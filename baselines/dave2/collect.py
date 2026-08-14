"""
Expert Data Collection for Behavioral Cloning

Collects (image, action) pairs from an expert controller driving in simulation.
The collected data is saved in a format compatible with DrivingDataset for training DAVE-2.

Expert sources:
    1. "scripted" - PD controller following a known reference path.
       This is the default and most reproducible option. The path is defined by waypoints in the Isaac Sim scene.
    2. "teleop" - Human driving via keyboard (FUTURE: or bluetooth controller).
       Higher quality demonstrations but harder to reproduce.
    3. "pid" - PID controller tracking lane center from lane_detector.
       Self-referential (uses our own CV module) but demonstrates what perfect lane-following looks like.

Output format:
    {output_dir}/
    |-- metadata.yaml        # Collection config + statistics
    |-- frames/
    |   |-- frame_000000.png # Saved at collection_hz rate
    |   |-- frame_000001.png
    |   |-- ...
    |-- labels.csv           # frame_id, steering, throttle, brake, speed

The collector saves at a fixed rate (default 10 Hz) independent of the sim's physics rate. This ensures consistent
temporal spacing in the dataset regardless of sim performance.

Enviroment Decoupling:
    The core collection functions (collect_from_gym_env, collect_teleop_from_gym_env) accept ANY Gymnasium
    environment that follows the Dict observation contract:
        obs["image"] -> (H, W, 3) uint8 RGB
        obs["vec"]   -> (12,) float32 telemetry

    This means the same collector works with:
        - IsaacROS2Env (current ROS2-couple env, for quick testing)
        - Future direct-API Isaac Sim env (BaseSimEnv adapter)
        - Future CARLA env
        - Any mock/test env that follows the contract

Usage:
    # Collect 5 minutes of scripted expert data:
    python -m baselines.dave2.collect \\
        --output data/expert_001 \\
        --episodes 20 \\
        --duration 300 \\
        --expert scripted

    # Collect with keyboard teleop:
    python -m baselines.dave2.collect \\
        --output data/expert_teleop \\
        --episodes 10 \\
        --expert teleop

Keyboard Controls (--expert teleop):

    |===================================================|
    | W / up arrow        Throttle (increase while held)|
    | S / down arrow      Brake (increase while held)   |
    | A / left arrow      Steer left                    |
    | D / right arrow     Steer right                   |
    | SPACE               Emergency brake (full stop)   |
    | R                   Reset steering to center      |
    | Q                   Quit collection and save      |
    | P                   Pause/resume recording        |
    |===================================================|

    Controls use smooth ramping: holding a key gradually increases the command value,
                                 releasing it decays back to zero.
    This produces smoother demonstrations than bang-bang keyboard input.

    Steering: ramps at 2.0/sec while held, decays at 3.0/sec on release.
    Throttle: ramps at 1.5/sec while held, decays at 2.0/sec on release.
    Brake:    applied instantly, decays at 2.0/sec on release

    The terminal must be in focus for key capture. The collector uses raw terminal mode (termios) on Linux
    - your terminal will be restored to normal on exit or Ctrl+C.

    Tips for good teleop data:
    - Drive at a steady moderate speed (1-2 m/s)
    - Make smooth turns, avoid jerky corrections
    - Include recovery maneuvers (drift off-center, then correct)
    - COllect at least 10-15 minutes for reasonable BC training

Dependencies:
    - OpenCV for image saving
    - NumPy
    - PyYAML (for metadata)
    - termios + tty (Linux stdlib, for keyboard teleop)
    - Gymnasium-compatible environment (passed in by caller)

Author: Aaron Hamil
Date: 03/02/26
Updated: 03/03/26
"""

import argparse
import csv
import time
import logging
import threading
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import numpy as np

from .experts import ScriptedExpert, KeyboardExpert
from .data_collector import DataCollector

logger = logging.getLogger(__name__)

#=================================================================================#
#               ENV-AGNOSTIC COLLECTION FUNCTIONS                                 #
# These accept ANY Gymnasium env with Dict obs {"image": ..., "vec": ...}.        #
# No ROS2 imports. No simulator-specific code. The CLI wires in the concrete env. #
#=================================================================================#=======#

def collect_from_gym_env(
    env,
    output_dir: str,
    expert=None,
    num_episodes: int = 20,
    max_steps_per_episode: int = 300,
    collection_hz: int = 10,
):
    """
    Collect scripted expert demonstrations from any Gymnasium environment.

    This is the core collection loop, fully decoupled from any specific simulator.
    It works with any env that provides Dict observations with 'image' and 'vec' keys.

    Args:
        env: Gymnasium environment instance. Must provide:
            obs["image"] -> (H, W, 3) uint8 RGB
            obs["vec"]   -> (12,) float32 telemetry
            env.step(action) -> standard Gymnasium 5-tuple
            env.reset()  -> (obs, info)
        output_dir: Where to save the dataset.
        expert: Expert controller with compute_action(telemetry) -> action.
                Defaults to ScriptedExpert().
        num_episodes: Number of driving episodes to collect.
        max_steps_per_episode: Maximum number of steps per episode.
        collection_hz: Frames per second to record (for metadata only - actual rate is determined by env.step() speed).
    """
    if expert is None:
        expert = ScriptedExpert()

    collector = DataCollector(
        output_dir=output_dir,
        collection_hz=collection_hz,
        expert=expert,
    )

    total_frames = 0
    start_time = time.time()

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                # Expert computes action from telemetry, collector saves frame + action
                telemetry = obs["vec"]
                action = collector.collect_from_arrays(obs["image"], telemetry)

                # Step environment with expert action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                total_frames += 1

            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"{step} steps, {total_frames} total frames"
            )

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user.")
    finally:
        duration = time.time() - start_time
        collector.save_metadata(duration=duration)
        collector.close()

    logger.info(
        f"Collected {total_frames} frames across {num_episodes} episodes "
        f"in {duration:.1f}s"
    )


def collect_teleop_from_gym_env(
    env,
    output_dir: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 300,
    collection_hz: int = 10,
):
    """
    Collect human expert demonstrations via keyboard teleop from any Gymnasium environment.

    Same contract as collect_from_gym_env but uses KeyboardExpert for human input.

    A status HUD is printed each step showing current steering, throttle, and brake values.
    Press P to pause recording (the vehicle keeps driving but frames aren't saved).
    Press Q to quit and save the dataset.

    Args:
        env: Gymnasium environment instance (same contract as collect_from_gym_env).
        output_dir: Where to save the dataset.
        num_episodes: Number of driving episodes to collect.
        max_steps_per_episode: Maximum steps per episode.
        collection_hz: Frames per second to record.
    """
    expert = KeyboardExpert(step_dt=1.0 / collection_hz)
    collector = DataCollector(output_dir=output_dir, collection_hz=collection_hz)

    total_frames = 0
    start_time = time.time()

    # Start keyboard listener (switches terminal to raw mode)
    expert.start()

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            print(f"\n Episode {episode + 1}/{num_episodes}")

            while not done and step < max_steps_per_episode:
                # Check if human pressed Q
                if expert.quit_requested:
                    logger.info("Quit requested - saving and exiting.")
                    break

                # Get human action from keyboard state
                telemetry = obs["vec"]
                action = expert.compute_action(telemetry)

                # Record frame (unless paused)
                if not expert.is_paused:
                    collector.collect_from_arrays_with_action(
                        obs["image"], telemetry, action
                    )
                    # Override recorded action with human's action
                    # (the collector calls expert.compute_action internally
                    # via ScriptedExpert, but we want the human's action)
                    total_frames += 1

                # Print HUD status line (overwrites same line)
                hud = expert.status_line()
                print(f"\r  Ep{episode + 1} Step{step:4d} | {hud} | "
                      f"Frames: {total_frames}", end="", flush=True)

                # Step environment with human's action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            if expert.quit_requested:
                break

    except KeyboardInterrupt:
        print() # Newline after the HUD line
        logger.info("Collection interrupted by Ctrl+C.")
    finally:
        # CRITICAL: restore terminal before any other cleanup
        expert.stop()
        print() # Clean newline after raw mode

        duration = time.time() - start_time
        collector.save_metadata(duration=duration, expert_name="KeyboardExpert")
        collector.close()
        env.close()

    logger.info(
        f"Teleop collection complete: {total_frames} frames "
        f"in {duration:.1f}s"
    )

#===============================================================================#
#          CONVENIENCE: CREATE DEFAULT ENV                                      #
# The CLI uses this to create an IsaacROS2Env if no env is passed.              #
# This is the only place with a ROS2 import - isolated from the core functions. #
#===============================================================================#
def _create_default_env(max_steps_per_episode: int = 300, collection_hz: int = 10):
    """
    Create the default IsaacROS2Env for CLI usage.

    This is the only function in the file that imports ROS2. It exists as a convenience for the CLI entrypoint.
    When the abstract env layer (BaseSimEnv + registry) is built, this will be replaced with
    SimFactory.create(config.sim).

    Returns:
        A Gymnasium-compatible environment instance.
    """
    try:
        from isaac_ros2_env import IsaacROS2Env, IsaacROS2Config
    except ImportError:
        logger.error(
            "isaac_ros2_env not found. Make sure ROS2 environment "
            "is available and Isaac Sim is running.\n"
            "Alternatively, pass a Gymnasium env directly to "
            "collect_from_gym_env() or collect_teleop_from_gym_env()."
        )
        raise

    config = IsaacROS2Config(
        img_width=160,
        img_height=90,
        episode_timeout=max_steps_per_episode / collection_hz,
    )

    return IsaacROS2Env(config=config)


#================#
# CLI ENTRYPOINT #
#================#

def main():
    parser = argparse.ArgumentParser(
        description="Collect expert driving data for behavioral cloning"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of driving episodes to collect"
    )
    parser.add_argument(
        "--max-steps", type=int, default=300,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--hz", type=int, default=10,
        help="Collection rate in Hz"
    )
    parser.add_argument(
        "--expert", type=str, default="scripted",
        choices=["scripted", "teleop", "pid"],
        help="Expert controller type (see module docstring for keyboard controls)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create the default environment (ROS2-coupled, for now)
    # When BaseSimEnv + registry exists, this becomes:
    #  config = ExperimentConfig.load(args.config)
    #  env = SimFactory.create(config.sim)
    env = _create_default_env(
        max_steps_per_episode=args.max_steps,
        collection_hz=args.hz,
    )

    try:
        if args.expert == "scripted":
            collect_from_gym_env(
                env=env,
                output_dir=args.output,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                collection_hz=args.hz,
            )
        elif args.expert == "teleop":
            collect_teleop_from_gym_env(
                env=env,
                output_dir=args.output,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                collection_hz=args.hz,
            )
        elif args.expert == "pid":
            logger.error("PID expert requires lane_detector integration - not yet implemented")
            raise NotImplementedError(
                "PID expert needs lane_detector.py to provide real-time "
                "lateral offset for the PID conreoller. This will be added "
                "when the direct-API environment is implemented."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
