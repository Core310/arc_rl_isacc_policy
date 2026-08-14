import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrappers.intersection_reward_wrapper import (
    IntersectionRewardWrapper,
    IntersectionRewardConfig,
)
from tests.mock_env_utils import MockEnv, obs_with_speed


class TestRewardWrapperPerStep:
    """Per-step terms: brake incentive, running-line penalty."""

    def test_brake_incentive_fires_in_approach_zone(self):
        """Brake action in approach zone with detected line => positive shaping."""
        canned = [
            {
                "obs": obs_with_speed(0.5),
                "reward": 0.0,
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.8,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={"worker_state": "cruising",
                                                  "worker_substate": "none"})
        cfg = IntersectionRewardConfig(
            brake_incentive=0.25, running_line_penalty=0.0,  # isolate brake term
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        action = np.array([0.0, 0.0, 0.8], dtype=np.float32)  # 80% brake
        obs, reward, *_, info = wrapped.step(action)
        expected = 0.25 * 0.8
        assert info[wrapped.INFO_KEY] == pytest.approx(expected)

    def test_brake_incentive_doesnt_fire_outside_approach_zone(self):
        """Detection too far away (> approach_zone_m) => no brake shaping."""
        canned = [
            {
                "obs": obs_with_speed(0.5),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 2.5,  # beyond approach_zone=1.5
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        cfg = IntersectionRewardConfig(running_line_penalty=0.0)
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0

    def test_running_line_penalty_fires_when_moving_on_red(self):
        """go_signal=0 + moving above threshold => penalty."""
        canned = [
            {
                "obs": obs_with_speed(0.5),  # > violation_speed_mps=0.25
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "approaching",
                    "stop_line_detected": False,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=-1.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(-1.0)

    def test_per_step_terms_dont_fire_in_committed(self):
        """Per-step shaping is gated to DECIDING only."""
        canned = [
            {
                "obs": obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "traversing",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.5,
                    "stop_line_confidence": 0.9,
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned)
        wrapped = IntersectionRewardWrapper(env)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 0.5], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0


class TestRewardWrapperOneShots:
    """One-shot bonuses/penalties: stop proximity, exit validation."""

    def test_perfect_stop_bonus_at_line(self):
        """Stop within tolerance of the line fires perfect_stop_bonus once."""
        canned = [
            {
                "obs": obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.03,  # within 0.08 tolerance
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",  # prev tick
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            perfect_stop_bonus=10.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(10.0)

    def test_overshoot_penalty_larger_than_undershoot(self):
        """Same distance magnitude: overshoot penalty > undershoot penalty."""
        canned_over = [
            {
                "obs": obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": -0.2,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]
        canned_under = [
            {
                "obs": obs_with_speed(0.05),
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.2,
                    "stop_line_confidence": 0.9,
                    "go_signal": 0.0,
                    "intersection": "int_main",
                },
            },
        ]

        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            overshoot_weight=20.0, undershoot_weight=5.0,
            stop_tolerance_m=0.08,
        )

        def _run(canned_steps):
            env = MockEnv(canned_steps, canned_reset_info={
                "worker_state": "deciding",
                "worker_substate": "approaching",
                "intersection": "int_main",
            })
            w = IntersectionRewardWrapper(env, config=cfg)
            w.reset()
            _, _, *_, info = w.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            return info[w.INFO_KEY]

        over = _run(canned_over)
        under = _run(canned_under)
        assert over < under < 0, f"overshoot {over} should be more negative than undershoot {under}"

    def test_stop_bonus_fires_only_once(self):
        """
        Holding STOPPING for multiple ticks fires the bonus only on the
        APPROACHING -> STOPPING transition (first STOPPING tick).
        """
        stopping_info = {
            "worker_state": "deciding",
            "worker_substate": "stopping",
            "stop_line_detected": True,
            "stop_line_distance_m": 0.03,
            "stop_line_confidence": 0.9,
            "go_signal": 0.0,
            "intersection": "int_main",
        }
        canned = [
            {"obs": obs_with_speed(0.05), "info": stopping_info},
            {"obs": obs_with_speed(0.05), "info": stopping_info},
            {"obs": obs_with_speed(0.05), "info": stopping_info},
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            perfect_stop_bonus=10.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()

        deltas = []
        for _ in range(3):
            _, _, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            deltas.append(info[wrapped.INFO_KEY])

        assert deltas[0] == pytest.approx(10.0)
        assert deltas[1] == 0.0
        assert deltas[2] == 0.0

    def test_correct_exit_bonus(self):
        """TRAVERSING -> EXITED with exit_correct=True fires correct bonus."""
        canned = [
            {
                "obs": obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "exited",
                    "exit_correct": True,
                    "exited_road": "road_B",
                    "committed_exit_road": "road_B",
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "committed",
            "worker_substate": "traversing",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            correct_exit_bonus=10.0, wrong_exit_penalty=-15.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(10.0)

    def test_wrong_exit_penalty(self):
        """TRAVERSING -> EXITED with exit_correct=False fires penalty."""
        canned = [
            {
                "obs": obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "exited",
                    "exit_correct": False,
                    "exited_road": "road_C",
                    "committed_exit_road": "road_B",
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "committed",
            "worker_substate": "traversing",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(
            brake_incentive=0.0, running_line_penalty=0.0,
            correct_exit_bonus=10.0, wrong_exit_penalty=-15.0,
        )
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == pytest.approx(-15.0)


class TestRewardWrapperDisabled:
    """Wrapper respects enabled=False (no-op) and is safe with legacy Worker."""

    def test_disabled_is_no_op(self):
        """enabled=False adds zero reward regardless of info."""
        canned = [
            {
                "obs": obs_with_speed(0.05),
                "reward": 1.5,
                "info": {
                    "worker_state": "deciding",
                    "worker_substate": "stopping",
                    "stop_line_detected": True,
                    "stop_line_distance_m": 0.03,
                    "stop_line_confidence": 0.9,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "deciding",
            "worker_substate": "approaching",
            "intersection": "int_main",
        })
        cfg = IntersectionRewardConfig(enabled=False, perfect_stop_bonus=10.0)
        wrapped = IntersectionRewardWrapper(env, config=cfg)
        wrapped.reset()
        _, reward, *_, info = wrapped.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert reward == 1.5, "base reward unchanged"
        assert info[wrapped.INFO_KEY] == 0.0

    def test_legacy_worker_info_produces_no_shaping(self):
        """Worker in legacy mode never produces APPROACHING/STOPPING/EXITED
        substates, so no one-shot fires."""
        canned = [
            {
                "obs": obs_with_speed(1.0),
                "info": {
                    "worker_state": "committed",
                    "worker_substate": "none",  # legacy Worker
                    "go_signal": 1.0,
                    "intersection": "int_main",
                },
            },
        ]
        env = MockEnv(canned, canned_reset_info={
            "worker_state": "cruising",
            "worker_substate": "none",
        })
        wrapped = IntersectionRewardWrapper(env)
        wrapped.reset()
        _, _, *_, info = wrapped.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        assert info[wrapped.INFO_KEY] == 0.0
