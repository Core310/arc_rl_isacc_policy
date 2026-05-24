"""
Waypoint Tracking Wrapper (Isaac Sim Adaptation)

Tracks the vehicle's trajectory via dead reckoning for self-supervised waypoint prediction learning.
The planning head in the hierarchical policy predicts future waypoints, and this wrapper records where the
vehicle actually went, providing supervision targets without any human labels.

Key features:
    1. Safety backfill: Marks last N steps as unsafe when crash occurs, so the repulsions loss can teach the
       planner to avoid crash paths.
    2. Global TrajectoryStore singleton: Allows the training loop to access trajectory data that would otherwise
       be lost in SB3's rollout buffer.
    3. Dead-reckoning position estimation from speed + yaw_rate.
    4. Proper episode boundary handling.

Adaptation from Unity version:
    * Removed Unity ground-truth position/yaw feedback (info['car_position']).
    * Dead reckoning uses speed * dt instead of per-step ds delta, because isaac_ros2_env[11] is cumulative
      total_distance, not ds.
    * Removed Unity visualization forwarding (set_waypoints on env).

Used by:
    train_policy_ros2.py (wraps IsaacROS2Env in make_env())
    losses/waypoint_losses.py (reads from TrajectoryStore for aux loss)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
import threading

class TrajectoryStore:
    """
    Thread-safe global store for trajectory data.

    Problem: SB3's rollout buffer discards episode info dicts after each rollout. But the waypoint auxiliary
             loss needs full trajectory from the just-completed episode to compute supervision targets.

    Solution: This singleton stores trajectory data per-environment, accessible from both the wrapper (which writes)
              and the custom PPO training loop (which reads for loss computation).

    Thread safety: Required because SB3 may reset environments from different threads during async collection.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._trajectories: Dict[int, Dict] = {}
        self._episode_safety: Dict[int, np.ndarray] = {}
        self._data_lock = threading.Lock()

    def store_trajectory(self, env_id: int, trajectory: Dict, safety_mask: np.ndarray):
        """Store trajectory data for an environment."""
        with self._data_lock:
            self._trajectories[env_id] = {
                "positions": trajectory["positions"].copy(),
                "yaws": trajectory["yaws"].copy(),
                "speeds": trajectory["speeds"].copy(),
            }
            self._episode_safety[env_id] = safety_mask.copy()

    def get_trajectory(self, env_id: int) -> Optional[Dict]:
        """Get stored trajectory for an environment."""
        with self._data_lock:
            return self._trajectories.get(env_id)

    def get_safety_mask(self, env_id: int) -> Optional[np.ndarray]:
        """Get safety mask for an environment."""
        with self._data_lock:
            return self._episode_safety.get(env_id)

    def get_full_trajectory(self, env_id: int) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Atomic read of both trajectory and safety mask to prevent race conditions."""
        with self._data_lock:
            traj = self._trajectories.get(env_id)
            mask = self._episode_safety.get(env_id)
            if traj is not None and mask is not None:
                return traj.copy(), mask.copy()
            return None, None

    def clear(self, env_id: Optional[int] = None):
        """Clear stored data."""
        with self._data_lock:
            if env_id is not None:
                self._trajectories.pop(env_id, None)
                self._episode_safety.pop(env_id, None)
            else:
                self._trajectories.clear()
                self._episode_safety.clear()

# Global instance
_trajectory_store = TrajectoryStore()


def get_trajectory_store() -> TrajectoryStore:
    """Get the global trajectory store instance."""
    return _trajectory_store


class WaypointTrackingWrapper(gym.Wrapper):
    """
    Wrapper that tracks:
        1. Predicted waypoints from policy (set externally via set_predicted_waypoints)
        2. Actual trajectory via dead reckoning (for self-supervised learning)
        3. Safety flags with backfill for crash trajectories

    Vectorized to handle multi-environment setups.
    """

    # Steps to backfill as unsafe when crash occurs (~0.5s at 20Hz)
    SAFETY_BACKFILL_STEPS = 10

    # Physics timestep (must match ARCProEnvCfg)
    # Env DT = sim_dt (0.002) * decimation (10) = 0.02
    DT = 0.02 # 50 Hz

    # Telemetry indices
    IDX_SPEED = 3
    IDX_YAW_RATE = 4

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        
        # Vectorized buffers
        self.position_history = [[] for _ in range(self.num_envs)]
        self.yaw_history = [[] for _ in range(self.num_envs)]
        self.speed_history = [[] for _ in range(self.num_envs)]
        self.safety_history = [[] for _ in range(self.num_envs)]
        
        # Dead-reckoning state (Vectorized)
        self.device = getattr(env, "device", "cpu")
        self._estimated_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._estimated_yaw = np.zeros(self.num_envs, dtype=np.float32)

        # Global store
        self._store = get_trajectory_store()

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and clear trajectory buffers."""
        for i in range(self.num_envs):
            self.position_history[i] = []
            self.yaw_history[i] = []
            self.speed_history[i] = []
            self.safety_history[i] = []
            self._store.clear(i)

        self._estimated_pos.fill(0.0)
        self._estimated_yaw.fill(0.0)

        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            return ret
        return ret, {}

    def step(self, action):
        """Step environment and record trajectory data."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert to numpy for easier processing
        reward_np = reward.cpu().numpy() if hasattr(reward, "cpu") else np.array(reward)
        term_np = terminated.cpu().numpy() if hasattr(terminated, "cpu") else np.array(terminated)
        trunc_np = truncated.cpu().numpy() if hasattr(truncated, "cpu") else np.array(truncated)
        
        # Update dead-reckoning position estimate (Vectorized)
        self._update_position_estimate(obs)

        # Extract telemetry
        if isinstance(obs, dict):
            vec = obs.get("policy", obs.get("vec"))
            speed_np = vec[:, self.IDX_SPEED].cpu().numpy()
        else:
            speed_np = np.zeros(self.num_envs)

        # Store data for each environment
        for i in range(self.num_envs):
            self.position_history[i].append(self._estimated_pos[i].copy())
            self.yaw_history[i].append(self._estimated_yaw[i])
            self.speed_history[i].append(speed_np[i])
            self.safety_history[i].append(1.0) # Initially safe

            # Handle episode end
            if term_np[i] or trunc_np[i]:
                self._handle_episode_end(i, term_np[i], trunc_np[i], reward_np[i])

        return obs, reward, terminated, truncated, info

    def _update_position_estimate(self, obs):
        """Dead-reckoning position update (Vectorized)."""
        if not isinstance(obs, dict): return

        vec = obs.get("policy", obs.get("vec"))
        if vec is None: return
        
        speed = vec[:, self.IDX_SPEED].cpu().numpy()
        yaw_rate = vec[:, self.IDX_YAW_RATE].cpu().numpy()

        # Update yaw
        self._estimated_yaw += yaw_rate * self.DT
        self._estimated_yaw = (self._estimated_yaw + np.pi) % (2 * np.pi) - np.pi

        # Update position
        # Isaac Sim: X = Forward, Y = Lateral, Z = Up
        # Standard navigation: yaw=0 points along X+
        ds = speed * self.DT
        self._estimated_pos[:, 0] += ds * np.cos(self._estimated_yaw) # Forward (X)
        self._estimated_pos[:, 1] += ds * np.sin(self._estimated_yaw) # Lateral (Y)
        # self._estimated_pos[:, 2] remains 0.0 (height)

    def _handle_episode_end(self, env_id: int, done: bool, truncated: bool, final_reward: float):
        """Handle episode end and safety backfill for a specific environment."""
        is_crash = done and not truncated and final_reward < 0

        if is_crash:
            num_steps = len(self.safety_history[env_id])
            backfill_start = max(0, num_steps - self.SAFETY_BACKFILL_STEPS)
            for j in range(backfill_start, num_steps):
                self.safety_history[env_id][j] = 0.0

        # Store in global store
        trajectory_data = {
            "positions": np.array(self.position_history[env_id]),
            "yaws": np.array(self.yaw_history[env_id]),
            "speeds": np.array(self.speed_history[env_id]),
        }
        safety_mask = np.array(self.safety_history[env_id])
        self._store.store_trajectory(env_id, trajectory_data, safety_mask)

        # Reset local buffers for next episode
        self.position_history[env_id] = []
        self.yaw_history[env_id] = []
        self.speed_history[env_id] = []
        self.safety_history[env_id] = []
        self._estimated_pos[env_id].fill(0.0)
        self._estimated_yaw[env_id] = 0.0
