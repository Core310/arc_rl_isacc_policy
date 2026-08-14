from dataclasses import dataclass, field
from typing import List, Optional
from agent.intersection_geometry import IntersectionLayout
from agent.stop_line_detector import StopLineDetectorConfig

# Telemetry Indices
IDX_TURN_TOKEN = 0     # Discrete turn command from Worker
IDX_GO_SIGNAL = 1      # Go/wait from Scheduler
IDX_GOAL_DIST = 2      # Zero-padded (PVP)
IDX_SPEED = 3
IDX_YAW_RATE = 4
IDX_LAST_STEER = 5
IDX_LAST_THROTTLE = 6
IDX_LAST_BRAKE = 7
IDX_LAT_ERR = 8        # Lateral error (m)
IDX_HDG_ERR = 9        # Heading error (rad)
IDX_KAPPA = 10         # Zero-padded (PVP)
IDX_DIST = 11

@dataclass
class WorkerConfig:
    """Configuration for the Worker node inside an Agent."""
    mode: str = "random"                            # "route" | "random" | "curriculum"
    route: List[int] = field(default_factory=list)  # Pre-planned turns
    curriculum_straight_bias: float = 0.6           # Initial bias toward STRAIGHT
    curriculum_decay_steps: int = 100_000           # Steps to reach uniform sampling
    intersection_cooldown: float = 3.0              # Seconds before re-triggering
                                                    # same intersection

    # Stop-line behavior (defaults ON — see use_stop_line docstring)
    use_stop_line: bool = True
    """
    If True, DECIDING is no longer zero-duration: the Worker holds
    go_signal=0 while approaching the intersection, the brake override
    forces a stop at the line, the Scheduler's go_signal releases
    traversal, and COMMITTED -> CRUISING validates the exit road.

    If False, legacy behavior: DECIDING flips to COMMITTED in one tick,
    no stop-line stop, no exit validation.

    Defaults ON. Flip to False only to reproduce legacy runs.
    """

    layout: IntersectionLayout = field(default_factory=IntersectionLayout)
    """Physical layout (lane widths, stop-line offset, pre-gate distance)."""

    detector_kind: str = "geometric"
    """
    Which stop-line detector to use when use_stop_line=True:
        "visual"    — classical CV on camera image. Deployment-realizable.
                      Requires an image kwarg on every WorkerNode.step call.
        "geometric" — privileged world-frame bootstrap. Training-only.
                      Use this until the visual pipeline is validated on
                      scene; then flip to "visual" via config.
    """

    detector_config: StopLineDetectorConfig = field(default_factory=StopLineDetectorConfig)
    """Thresholds and camera intrinsics for the visual detector."""

    # Substate timing (all in seconds)
    stop_dwell_time: float = 0.5
    """How long the agent must be stopped at the line before the
    Worker releases from STOPPING toward CLEARED. Gives the Scheduler
    a moment to issue a real go/wait decision."""

    stopped_speed_threshold: float = 0.1
    """Speed (m/s) below which the agent is considered stopped."""

    moving_speed_threshold: float = 0.25
    """Speed (m/s) above which the agent is considered moving. Used
    for COMMITTED -> EXITED transition hysteresis."""

    # Planar path planner (intersection traversal reference)

    plan_exit_ahead_m: float = 1.5
    """How far past the exit-road entry to extend the final plan
    waypoint, in meters. ~1-2 F1TENTH car lengths. Gives downstream
    reward shaping / scheduler overlap detection a runway past the
    intersection box on the exit road."""



@dataclass
class AgentConfig:
    """Configuration for one Agent."""
    agent_id: str = "agent_0"
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    brake_decel: float = 0.8

