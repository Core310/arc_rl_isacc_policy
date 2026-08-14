from __future__ import annotations
import math
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    TurnCommand,
    ExitOption,
)
from agent.intersection_geometry import (
    IntersectionLayout,
    approach_axes,
    infer_current_approach,
    detect_exited_road,
    within_pre_gate_planar,
)
from agent.stop_line_detector import (
    StopLineDetection,
    StopLineDetectionContext,
    StopLineDetectorConfig,
    StopLineDetectorBase,
    make_stop_line_detector,
)
from agent.planar_planner import (
    PlanarPathPlanner,
    PlanarPath,
)
from agent.config import WorkerConfig

logger = logging.getLogger(__name__)

class WorkerNode:
    """
    Route planning node inside an Agent.

    Decides WHICH direction to go at intersections. Does NOT control
    the vehicle, that's Main's job. The Worker emits a discrete
    turn_token that gets injected into the observation vector for
    the policy to read.

    Primary state machine:
        CRUISING  -> (pre-gate armed) -> DECIDING
        DECIDING  -> (stopped + go) -> COMMITTED
        COMMITTED -> (exited correctly) -> CRUISING

    Orthogonal substate (meaningful only inside DECIDING/COMMITTED,
    populated when use_stop_line=True):
        DECIDING:
            APPROACHING — moving toward the line, brake override held
            STOPPING    — stopped at the line, waiting for Scheduler go
            CLEARED     — scheduler said go, release into COMMITTED next tick
        COMMITTED:
            TRAVERSING  — inside the intersection box
            EXITED      — cleared the exit side, ready to go CRUISING

    Legacy behavior (use_stop_line=False):
        DECIDING flips to COMMITTED in one tick. No stop. No exit
        validation. Preserved for reproducing legacy runs.

    The Worker holds its last committed turn_token until the agent
    leaves the intersection, so the policy has a stable signal
    throughout the maneuver.
    """

    # Worker states
    CRUISING = "cruising"       # On open road, no decision needed
    DECIDING = "deciding"       # At intersection, picking a turn / stopping
    COMMITTED = "committed"     # Turn chosen, traversing intersection

    # Substates (only meaningful when use_stop_line=True)
    SUB_NONE = "none"
    SUB_APPROACHING = "approaching"
    SUB_STOPPING = "stopping"
    SUB_CLEARED = "cleared"
    SUB_TRAVERSING = "traversing"
    SUB_EXITED = "exited"

    def __init__(
        self,
        agent_id: str,
        graph: IntersectionGraph,
        config: Optional[WorkerConfig] = None,
    ):
        self.agent_id = agent_id
        self.graph = graph
        self.config = config or WorkerConfig()

        # Primary state
        self._state = self.CRUISING
        self._substate: str = self.SUB_NONE
        self._current_intersection: Optional[str] = None
        self._turn_token: int = TurnCommand.STRAIGHT
        self._go_signal: float = 1.0   # Default: go (no scheduler block)
        self._route_index: int = 0
        self._total_steps: int = 0

        # Cooldown: prevent re-triggering the same intersection
        # immediately after leaving it
        self._last_intersection_id: Optional[str] = None
        self._cooldown_remaining: float = 0.0

        # Intersection approach context (populated on CRUISING -> DECIDING)
        self._current_approach_road_id: Optional[str] = None
        self._stop_dwell_elapsed: float = 0.0

        # Exit bookkeeping (populated on turn commit)
        self._committed_exit_road_id: Optional[str] = None
        self._exited_road_id: Optional[str] = None
        self._exit_correct: Optional[bool] = None

        # Detector + last detection (only populated when use_stop_line=True)
        self._detector: Optional[StopLineDetectorBase] = None
        if self.config.use_stop_line:
            self._detector = make_stop_line_detector(
                kind=self.config.detector_kind,
                config=self.config.detector_config,
                layout=self.config.layout,
            )
        self._last_detection: StopLineDetection = StopLineDetection()
        self._warned_missing_image: bool = False

        # Planar path planner (intersection traversal reference)
        # Generates a world-frame PlanarPath for each traversal. The
        # plan is stored on this Worker and exposed via
        # AgentNode.current_plan; it never enters the policy's
        # observation vector (PVP).
        self._path_planner = PlanarPathPlanner(
            exit_plan_ahead_m=self.config.plan_exit_ahead_m,
        )
        self._current_plan: Optional[PlanarPath] = None

    @property
    def turn_token(self) -> int:
        """Current turn command for the policy."""
        return self._turn_token

    @property
    def go_signal(self) -> float:
        """Current go/wait signal from Scheduler."""
        return self._go_signal

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_intersection_id(self) -> Optional[str]:
        return self._current_intersection

    @property
    def substate(self) -> str:
        """Current substate (meaningful only with use_stop_line=True)."""
        return self._substate

    @property
    def last_detection(self) -> StopLineDetection:
        """Most recent stop-line detection result (cached)."""
        return self._last_detection

    @property
    def committed_exit_road_id(self) -> Optional[str]:
        """Road the turn_token commits the agent to entering.
        None outside DECIDING/COMMITTED."""
        return self._committed_exit_road_id

    @property
    def exited_road_id(self) -> Optional[str]:
        """Road the agent actually exited onto (set once on exit)."""
        return self._exited_road_id

    @property
    def exit_correct(self) -> Optional[bool]:
        """True if exited road matches committed intent. None until exit."""
        return self._exit_correct

    @property
    def current_approach_road_id(self) -> Optional[str]:
        """Road the agent is on inside the current DECIDING/COMMITTED cycle."""
        return self._current_approach_road_id

    @property
    def current_plan(self) -> Optional[PlanarPath]:
        """Active planar reference path through the current intersection,
        or None when CRUISING / between traversals. Not observable by
        the policy (PVP)."""
        return self._current_plan

    def step(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float = 1.0 / 10.0,
        scheduler=None,
        image: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """
        Worker step using GLOBAL position (PhysX ground truth).

        Called every environment step by AgentNode.step(). The active
        path uses global (x, y) and a planar pre-gate against the
        intersection center. The EKF-native step_topological() path is
        shelved on branch `legacy/frenet-topological`.

        Args:
            position: (x, y) in world frame from PhysX.
            heading: Agent heading in radians.
            speed: Agent speed in m/s.
            dt: Time since last step (for cooldown decay and substate timing).
            scheduler: Optional WorkerScheduler for multi-agent.
            image: Forward camera image (H, W, 3) uint8 for the visual
                stop-line detector. Ignored when use_stop_line=False or
                detector_kind="geometric". Pass None to fall back to
                geometric detection (training only).

        Returns:
            (turn_token, go_signal) to inject into vec[0:2].
        """
        self._total_steps += 1

        # Decay cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)

        # Legacy path (zero-duration DECIDING, no stop-line behavior)
        if not self.config.use_stop_line:
            self._step_legacy(position, heading, speed, scheduler)
            return self._turn_token, self._go_signal

        # Stop-line-enabled path
        self._step_with_stop_line(position, heading, speed, dt, scheduler, image)
        return self._turn_token, self._go_signal

    def _step_legacy(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        scheduler,
    ) -> None:
        """
        Original CRUISING/DECIDING/COMMITTED flow (use_stop_line=False).

        Preserved verbatim so legacy experiments reproduce. The
        only change is that go_signal is still managed the same way,
        the new substate field stays SUB_NONE throughout.
        """
        x, y = position
        intersection = self.graph.nearest_intersection(x, y)

        if self._state == self.CRUISING:
            if intersection is not None:
                if (
                    intersection.node_id == self._last_intersection_id
                    and self._cooldown_remaining > 0
                ):
                    pass  # Still in cooldown, stay CRUISING
                else:
                    self._state = self.DECIDING
                    self._current_intersection = intersection.node_id
                    self._decide_turn(intersection, heading, speed, position, scheduler)
                    self._state = self.COMMITTED

        elif self._state == self.COMMITTED:
            if intersection is None:
                self._last_intersection_id = self._current_intersection
                self._cooldown_remaining = self.config.intersection_cooldown
                self._current_intersection = None
                self._state = self.CRUISING
            else:
                if scheduler is not None:
                    self._go_signal = scheduler.query_go_signal(
                        self.agent_id,
                        self._current_intersection,
                        self._turn_token,
                        position,
                        heading,
                        speed,
                    )

        if self._state == self.CRUISING:
            self._go_signal = 1.0

    def _step_with_stop_line(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float,
        scheduler,
        image: Optional[np.ndarray],
    ) -> None:
        """
        Extended flow: pre-gate arms detector, DECIDING waits for stop,
        COMMITTED validates exit road.

        Primary state transitions happen at substate boundaries:
            APPROACHING -> STOPPING          (stopped near the line)
            STOPPING    -> CLEARED           (scheduler released go_signal)
            CLEARED     -> ...               (next tick: DECIDING -> COMMITTED)
            TRAVERSING  -> EXITED            (exit_road detected)
            EXITED      -> ...               (next tick: COMMITTED -> CRUISING)
        """
        x, y = position

        # Nearest intersection (still used for CRUISING trigger — the
        # pre-gate arming logic also uses it, but nearest_intersection
        # gives us the node id cheaply).
        intersection = self.graph.nearest_intersection(x, y)

        if self._state == self.CRUISING:
            self._handle_cruising(intersection, position, heading, speed, dt, scheduler, image)

        elif self._state == self.DECIDING:
            self._handle_deciding(position, heading, speed, dt, scheduler, image)

        elif self._state == self.COMMITTED:
            self._handle_committed(position, heading, speed, dt, scheduler)

        # Sticky go_signal default for pure CRUISING (no pre-gate armed)
        if self._state == self.CRUISING and self._substate == self.SUB_NONE:
            self._go_signal = 1.0

    ### Primary-state handlers

    def _handle_cruising(
        self,
        intersection: Optional[IntersectionNode],
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float,
        scheduler,
        image: Optional[np.ndarray],
    ) -> None:
        """
        CRUISING: decide whether to promote to DECIDING this tick.

        Trigger: agent is within pre_gate_distance of an intersection
        along its current approach edge AND cooldown has expired. We
        use the Frenet pre-gate (arc-length based) rather than the
        legacy 2D radius so long approaches don't misfire.
        """
        if intersection is None:
            return

        # Cooldown
        if (
            intersection.node_id == self._last_intersection_id
            and self._cooldown_remaining > 0
        ):
            return

        # Identify the approach the agent is on
        approach = infer_current_approach(position, heading, intersection, self.graph)
        if approach is None:
            return
        road_id, approach_info = approach

        # Planar pre-gate: arm iff agent is within pre_gate_distance of
        # the intersection center along the approach axis. No Frenet,
        # no arc-length, no edge geometry lookup — the JSON topology
        # already gives us the intersection position and approach
        # heading, which is everything we need.
        if intersection.position is None:
            # Graph not calibrated for this node — can't gate. Bail.
            return
        if not within_pre_gate_planar(
            position,
            intersection.position,
            approach_info.heading_rad,
            self.config.layout,
        ):
            return

        # Promote to DECIDING. Pick turn now so turn_token is stable
        # for the entire stop phase. go_signal will be held at 0 until
        # STOPPING -> CLEARED.
        self._state = self.DECIDING
        self._substate = self.SUB_APPROACHING
        self._current_intersection = intersection.node_id
        self._current_approach_road_id = road_id
        self._stop_dwell_elapsed = 0.0
        self._decide_turn(intersection, heading, speed, position, scheduler)

        # Cache the committed exit road (for exit validation later)
        if self._turn_token in approach_info.exits:
            self._committed_exit_road_id = approach_info.exits[self._turn_token].exit_road_id
        else:
            self._committed_exit_road_id = None

        # Generate the planar reference path for this traversal.
        # Stored on the Worker and exposed via AgentNode.current_plan.
        # Privileged: uses ground-truth position + topology, never
        # enters the observation vector. None is allowed — downstream
        # consumers (info dict, scheduler, future reward shaping)
        # must tolerate absent plans.
        self._current_plan = self._path_planner.plan(
            current_xy=position,
            current_heading=heading,
            intersection=intersection,
            entry_road_id=road_id,
            exit_road_id=self._committed_exit_road_id,
            turn_command=self._turn_token,
            layout=self.config.layout,
        )

        # During APPROACHING we override go_signal to 0 regardless of
        # scheduler, so the brake override kicks in.
        self._go_signal = 0.0

        # Run detector on this first DECIDING tick
        self._run_detector(position, heading, intersection, image)

    def _handle_deciding(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float,
        scheduler,
        image: Optional[np.ndarray],
    ) -> None:
        """
        DECIDING: APPROACHING -> STOPPING -> CLEARED -> COMMITTED.

        APPROACHING: brake override active (go=0). When agent slows
            near the line OR has crossed it while slow, promote to
            STOPPING.
        STOPPING: dwell until scheduler releases. While waiting, hold
            go_signal at whatever scheduler last said (may flip to 1
            early if this is the only agent at the intersection).
        CLEARED: promoted on the tick scheduler returns go_signal=1
            AND dwell has elapsed. Next tick rolls to COMMITTED.
        """
        intersection = None
        if self._current_intersection is not None:
            intersection = self.graph.get_intersection(self._current_intersection)

        # Run detector every DECIDING tick
        self._run_detector(position, heading, intersection, image)

        if self._substate == self.SUB_APPROACHING:
            # Stop trigger: either we're slow near the line, or we've
            # already coasted past it and should latch the stop anyway.
            det = self._last_detection
            near_line = det.detected and abs(det.distance_m) < (
                self.config.layout.stop_line_tolerance + 0.25
            )
            # Hard stop latch if speed is very low even without detection
            hard_stopped = speed < self.config.stopped_speed_threshold
            if (near_line and speed < self.config.stopped_speed_threshold) or hard_stopped:
                self._substate = self.SUB_STOPPING
                self._stop_dwell_elapsed = 0.0
            # go_signal stays 0 (brake override)
            self._go_signal = 0.0

        elif self._substate == self.SUB_STOPPING:
            self._stop_dwell_elapsed += dt

            # Query scheduler. If scheduler says go AND dwell has elapsed,
            # release.
            if scheduler is not None and self._current_intersection is not None:
                scheduler_go = scheduler.query_go_signal(
                    self.agent_id,
                    self._current_intersection,
                    self._turn_token,
                    position,
                    heading,
                    speed,
                )
            else:
                scheduler_go = 1.0

            if scheduler_go >= 0.5 and self._stop_dwell_elapsed >= self.config.stop_dwell_time:
                self._substate = self.SUB_CLEARED
                self._go_signal = 1.0
            else:
                # Still holding at the line — brake override remains
                self._go_signal = 0.0

        elif self._substate == self.SUB_CLEARED:
            # Transition to COMMITTED on the same tick — policy already
            # has go_signal=1 this step.
            self._state = self.COMMITTED
            self._substate = self.SUB_TRAVERSING
            self._go_signal = 1.0

    def _handle_committed(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float,
        scheduler,
    ) -> None:
        """
        COMMITTED: TRAVERSING -> EXITED -> CRUISING.

        TRAVERSING: inside the intersection. Keep scheduler updated
            (RVO may modify go_signal mid-maneuver for dynamic conflicts).
        EXITED: exit road detected. Compare with committed_exit_road_id
            to set exit_correct. Transition to CRUISING.
        """
        intersection = None
        if self._current_intersection is not None:
            intersection = self.graph.get_intersection(self._current_intersection)

        if intersection is None:
            # Something drifted; fall back to CRUISING defensively
            self._finish_intersection(exited_road=None)
            return

        # Check exit detection
        exited = detect_exited_road(
            position, heading, intersection, self.config.layout
        )

        if self._substate == self.SUB_TRAVERSING:
            if exited is not None and speed > self.config.moving_speed_threshold:
                self._substate = self.SUB_EXITED
                self._exited_road_id = exited
                self._exit_correct = (
                    self._committed_exit_road_id is not None
                    and exited == self._committed_exit_road_id
                )
            else:
                # Still inside — scheduler may modify go_signal for RVO
                if scheduler is not None:
                    self._go_signal = scheduler.query_go_signal(
                        self.agent_id,
                        self._current_intersection,
                        self._turn_token,
                        position,
                        heading,
                        speed,
                    )

        elif self._substate == self.SUB_EXITED:
            # One tick for observers to read exited_road / exit_correct
            # from info, then release.
            self._finish_intersection(exited_road=self._exited_road_id)

    def _finish_intersection(self, exited_road: Optional[str]) -> None:
        """Clean up and transition COMMITTED -> CRUISING."""
        self._last_intersection_id = self._current_intersection
        self._cooldown_remaining = self.config.intersection_cooldown
        self._current_intersection = None
        self._current_approach_road_id = None
        self._state = self.CRUISING
        self._substate = self.SUB_NONE
        # Keep turn_token stable until the next intersection triggers;
        # Main is still executing the maneuver geometrically.
        # exited_road / exit_correct / committed_exit_road_id stay
        # populated until next CRUISING -> DECIDING, so the wrapper's
        # info dict can surface them on this tick.

    ### Detector

    def _run_detector(
        self,
        position: Tuple[float, float],
        heading: float,
        intersection: Optional[IntersectionNode],
        image: Optional[np.ndarray],
    ) -> None:
        """Build detection context and run the configured detector."""
        if self._detector is None or intersection is None:
            self._last_detection = StopLineDetection()
            return

        # Resolve the approach heading for the current DECIDING cycle
        approach_heading = None
        if self._current_approach_road_id is not None:
            app = intersection.approaches.get(self._current_approach_road_id)
            if app is not None:
                approach_heading = app.heading_rad

        # Warn once if visual mode was requested but no image arrived
        if (
            self.config.detector_kind == "visual"
            and image is None
            and not self._warned_missing_image
        ):
            logger.warning(
                "[%s] detector_kind='visual' but no image provided; "
                "detection will return null until an image is supplied.",
                self.agent_id,
            )
            self._warned_missing_image = True

        ctx = StopLineDetectionContext(
            image=image,
            agent_xy=position,
            intersection_center=intersection.position,
            approach_heading_rad=approach_heading,
            active=True,
        )
        self._last_detection = self._detector.detect(ctx)

    def _decide_turn(
        self,
        intersection: IntersectionNode,
        heading: float,
        speed: float,
        position: Tuple[float, float],
        scheduler=None,
    ) -> None:
        """
        Pick a turn command based on the Worker's mode.

        Modifies self._turn_token and self._go_signal in place.
        """
        exits = self.graph.get_exit_options(intersection.node_id, heading)

        if not exits:
            # No recognized approach — default to straight
            logger.debug(
                f"[{self.agent_id}] No exits at {intersection.node_id} "
                f"for heading {math.degrees(heading):.0f}deg, defaulting STRAIGHT"
            )
            self._turn_token = TurnCommand.STRAIGHT
            return

        available_commands = [e.turn_command for e in exits]

        if self.config.mode == "route":
            self._turn_token = self._pick_from_route(available_commands)
        elif self.config.mode == "curriculum":
            self._turn_token = self._pick_curriculum(available_commands)
        else:  # "random"
            self._turn_token = self._pick_random(available_commands)

        # Register with scheduler for multi-agent coordination
        if scheduler is not None:
            self._go_signal = scheduler.register_intent(
                agent_id=self.agent_id,
                intersection_id=intersection.node_id,
                turn_command=self._turn_token,
                position=position,
                heading=heading,
                speed=speed,
            )
        else:
            self._go_signal = 1.0  # No scheduler = always go

        logger.debug(
            f"[{self.agent_id}] At {intersection.node_id}: "
            f"chose {TurnCommand.name(self._turn_token)} "
            f"(available: {[TurnCommand.name(c) for c in available_commands]}) "
            f"go={self._go_signal}"
        )

    def _pick_from_route(self, available: List[int]) -> int:
        """Pick next turn from pre-defined route sequence."""
        if not self.config.route:
            return TurnCommand.STRAIGHT

        cmd = self.config.route[self._route_index % len(self.config.route)]
        self._route_index += 1

        # If planned turn isn't available, fall back to straight or first available
        if cmd not in available:
            if TurnCommand.STRAIGHT in available:
                return TurnCommand.STRAIGHT
            return available[0]
        return cmd

    def _pick_random(self, available: List[int]) -> int:
        """Uniform random from available exits."""
        return int(np.random.choice(available))

    def _pick_curriculum(self, available: List[int]) -> int:
        """
        Weighted sampling: straight-biased early, uniform late

        Early training: agent mostly goes straight, learning basic
        lane following before tackling turns. As training progresses,
        the distribution shifts toward uniform sampling over all exits.
        """
        progress = min(1.0, self._total_steps / max(1, self.config.curriculum_decay_steps))
        straight_weight = self.config.curriculum_straight_bias * (1.0 - progress)

        weights = []
        for cmd in available:
            if cmd == TurnCommand.STRAIGHT:
                weights.append(1.0 + straight_weight)
            else:
                weights.append(1.0)

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        return int(np.random.choice(available, p=weights))

    def reset(self) -> None:
        """Reset Worker state for new episode (not training progress)."""
        self._state = self.CRUISING
        self._substate = self.SUB_NONE
        self._current_intersection = None
        self._turn_token = TurnCommand.STRAIGHT
        self._go_signal = 1.0
        self._last_intersection_id = None
        self._cooldown_remaining = 0.0

        # Stop-line / intersection bookkeeping
        self._current_approach_road_id = None
        self._stop_dwell_elapsed = 0.0
        self._committed_exit_road_id = None
        self._exited_road_id = None
        self._exit_correct = None
        self._last_detection = StopLineDetection()
        self._warned_missing_image = False

        # Planar path planner state
        self._current_plan = None

        # NOTE: _route_index and _total_steps persist across episodes
        # NOTE: _detector instance is intentionally NOT reset — it's
        # stateless across frames, no per-episode cleanup needed


