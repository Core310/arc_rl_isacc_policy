"""
Agent Node - Hierarchical Driver-Worker Architecture
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from agent.intersection_graph import IntersectionGraph, TurnCommand
from agent.stop_line_detector import StopLineDetection
from agent.config import AgentConfig, WorkerConfig, IDX_TURN_TOKEN, IDX_GO_SIGNAL, IDX_SPEED, IDX_YAW_RATE
from agent.worker_node import WorkerNode
from agent.main_node import MainNode

class AgentNode:
    """
    Top-level agent containing Worker and Main nodes.

    One AgentNode per vehicle. The environment creates AgentNodes and
    calls agent.step() each timestep, threading position data to the
    Worker and observation data to the Main.

    Integration with IsaacDirectEnv:
        The env calls:
            1. agent.worker_step(pos, heading, speed, dt, scheduler)
               -> gets (turn_token, go_signal)
            2. agent.prepare_obs(raw_obs)
               -> injects token + signal into obs
            3. Policy runs on prepared obs -> raw_action
            4. agent.apply_action_gate(raw_action)
               -> applies go/brake safety override
            5. env applies gated action to vehicle

    Integration with training loop (train_policy_ros2.py):
        The AgentNode doesn't interfere with SB3's training loop.
        It wraps the observation before SB3 sees it and gates the
        action after SB3 produces it. From SB3's perspective, the
        observation space is unchanged (still Dict with image + vec),
        and the action space is unchanged (still Box [steer, thr, brk]).
        The Worker's decisions appear as part of the observation.
    """

    def __init__(
        self,
        graph: IntersectionGraph,
        config: Optional[AgentConfig] = None,
        scheduler=None,
    ):
        """
        Args:
            graph: Shared IntersectionGraph (read-only).
            config: Agent configuration.
            scheduler: Optional WorkerScheduler for multi-agent.
        """
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id
        self.scheduler = scheduler

        # Internal nodes
        self.worker = WorkerNode(
            agent_id=self.agent_id,
            graph=graph,
            config=self.config.worker,
        )
        self.main = MainNode(
            agent_id=self.agent_id,
            brake_decel=self.config.brake_decel,
        )

        # Latest state for logging / debugging
        self._last_turn_token: int = TurnCommand.STRAIGHT
        self._last_go_signal: float = 1.0

    def worker_step(
        self,
        position: Tuple[float, float],
        heading: float,
        speed: float,
        dt: float = 0.1,
        image: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """
        Run the Worker node for this timestep.

        Queries the intersection graph, picks a turn if at intersection,
        coordinates with Scheduler if present.

        Args:
            position: (x, y) from PhysX ground truth or EKF.
            heading: Current heading in radians.
            speed: Current speed in m/s.
            dt: Time since last call.
            image: Forward camera image (H, W, 3) uint8 for the visual
                stop-line detector. Optional — pass None to let the
                Worker fall back to whatever its configured detector
                can do without an image (geometric mode ignores this).

        Returns:
            (turn_token, go_signal) to be injected into obs.
        """
        token, go = self.worker.step(
            position=position,
            heading=heading,
            speed=speed,
            dt=dt,
            scheduler=self.scheduler,
            image=image,
        )
        self._last_turn_token = token
        self._last_go_signal = go
        return token, go

    def prepare_obs(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Inject Worker's commands into observation for the policy.

        Call this AFTER worker_step() and BEFORE policy forward pass.
        """
        return self.main.prepare_observation(
            obs,
            turn_token=self._last_turn_token,
            go_signal=self._last_go_signal,
        )

    def apply_action_gate(self, action: np.ndarray) -> np.ndarray:
        """
        Apply go/brake safety override on policy's raw action.

        Call this AFTER policy produces action, BEFORE sending to vehicle.
        """
        return self.main.apply_go_brake_gate(action, self._last_go_signal)

    def reset(self) -> None:
        """Reset for new episode."""
        self.worker.reset()
        self._last_turn_token = TurnCommand.STRAIGHT
        self._last_go_signal = 1.0

    @property
    def current_plan(self):
        """Active planar reference path for this agent's current
        intersection traversal, or None when CRUISING.

        Exposed on AgentNode so the scheduler, reward wrapper, and
        future MARL coordination can read plans from every agent
        without reaching into WorkerNode internals. NOT in the
        observation vector (PVP)."""
        return self.worker.current_plan

    @property
    def info(self) -> Dict:
        """Current agent state for logging."""
        det = self.worker.last_detection
        plan = self.worker.current_plan
        return {
            "agent_id": self.agent_id,
            "worker_state": self.worker.state,
            "worker_substate": self.worker.substate,
            "turn_token": self._last_turn_token,
            "turn_name": TurnCommand.name(self._last_turn_token),
            "go_signal": self._last_go_signal,
            "intersection": self.worker.current_intersection_id,
            # Stop-line detector output (non-privileged)
            "stop_line_detected": bool(det.detected),
            "stop_line_distance_m": float(det.distance_m),
            "stop_line_confidence": float(det.confidence),
            "stop_line_source": det.source,
            # Intersection commitment + exit validation (non-privileged)
            "committed_exit_road": self.worker.committed_exit_road_id,
            "exited_road": self.worker.exited_road_id,
            "exit_correct": self.worker.exit_correct,
            "approach_road": self.worker.current_approach_road_id,
            # Planar path plan (non-privileged; for logging and future
            # reward shaping / scheduler overlap detection only — NOT
            # exposed in the observation vector).
            "plan_present": plan is not None,
            "plan_num_waypoints": plan.num_waypoints if plan is not None else 0,
            "plan_length_m": float(plan.length) if plan is not None else 0.0,
        }
