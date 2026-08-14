import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional

class MockEnv(gym.Env):
    """
    Minimal env that returns a canned sequence of (obs, reward, done,
    trunc, info) tuples. Lets us drive the IntersectionRewardWrapper
    through any scenario without needing the Worker or Isaac Sim.
    """

    def __init__(self, canned_steps: list, canned_reset_info: Optional[Dict] = None):
        super().__init__()
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
            "vec": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self._canned = list(canned_steps)
        self._reset_info = canned_reset_info or {}
        self._i = 0

    def _zero_obs(self) -> Dict[str, np.ndarray]:
        return {
            "image": np.zeros((90, 160, 3), dtype=np.uint8),
            "vec": np.zeros(12, dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        self._i = 0
        return self._zero_obs(), dict(self._reset_info)

    def step(self, action):
        if self._i >= len(self._canned):
            # Exhausted — return terminal no-op
            return self._zero_obs(), 0.0, True, False, {}
        step_data = self._canned[self._i]
        self._i += 1
        obs = step_data.get("obs", self._zero_obs())
        reward = step_data.get("reward", 0.0)
        terminated = step_data.get("terminated", False)
        truncated = step_data.get("truncated", False)
        info = step_data.get("info", {})
        return obs, reward, terminated, truncated, info


def obs_with_speed(speed: float) -> Dict[str, np.ndarray]:
    obs = {
        "image": np.zeros((90, 160, 3), dtype=np.uint8),
        "vec": np.zeros(12, dtype=np.float32),
    }
    obs["vec"][3] = speed
    return obs
