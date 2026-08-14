import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.intersection_graph import TurnCommand
from agent.agent_node import AgentNode, AgentConfig, WorkerConfig
from wrappers.intersection_reward_wrapper import (
    IntersectionRewardWrapper,
    IntersectionRewardConfig,
)
from tests.mock_env_utils import MockEnv, obs_with_speed

class TestFullStack:
    """
    Drive a real AgentNode + IntersectionRewardWrapper through a scripted
    trajectory. Asserts end-to-end reward accumulation matches expectation.
    """

    def test_nominal_approach_stop_exit_accumulates_reward(self, cross_graph, default_layout):
        """
        Scenario: east-bound on road_D, go STRAIGHT (→ road_B). Script
        a trajectory: cruise → pre-gate arm → slow → stop near line →
        dwell → release → traverse → exit correctly on road_B.
        """
        worker_cfg = WorkerConfig(
            use_stop_line=True, detector_kind="geometric",
            layout=default_layout,
            mode="route", route=[TurnCommand.STRAIGHT],
            stopped_speed_threshold=0.1,
            stop_dwell_time=0.1,
            moving_speed_threshold=0.2,
        )
        agent = AgentNode(
            graph=cross_graph,
            config=AgentConfig(agent_id="a0", worker=worker_cfg),
        )

        trajectory = [
            ((-2.0, 0.0), 0.0, 1.0, 0.05),
            ((-1.2, 0.0), 0.0, 0.8, 0.05),
            ((-0.9, 0.0), 0.0, 0.4, 0.05),
            ((-0.6, 0.0), 0.0, 0.05, 0.05),
            ((-0.6, 0.0), 0.0, 0.05, 0.1),
            ((-0.6, 0.0), 0.0, 0.05, 0.1),
            ((-0.3, 0.0), 0.0, 0.5, 0.05),
            ((0.3, 0.0),  0.0, 1.0, 0.05),
            ((1.0, 0.0),  0.0, 1.0, 0.05),
        ]

        canned_steps = []
        for pos, heading, speed, dt in trajectory:
            agent.worker_step(
                position=pos, heading=heading, speed=speed, dt=dt,
            )
            obs = obs_with_speed(speed)
            canned_steps.append({
                "obs": obs,
                "reward": 0.0,
                "info": dict(agent.info),
            })

        initial_info = {"worker_state": "cruising", "worker_substate": "none"}
        env = MockEnv(canned_steps, canned_reset_info=initial_info)
        cfg = IntersectionRewardConfig()
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        deltas = []
        for _ in range(len(trajectory)):
            action = np.array([0.0, 0.0, 0.8], dtype=np.float32)
            _, _, *_, info = wrapped.step(action)
            deltas.append(info[wrapped.INFO_KEY])

        assert any(d >= cfg.perfect_stop_bonus * 0.5 for d in deltas), \
            f"expected stop bonus spike in deltas={deltas}"

        final_info = canned_steps[-1]["info"]
        assert final_info["exited_road"] == "road_B"
        assert final_info["exit_correct"] is True
