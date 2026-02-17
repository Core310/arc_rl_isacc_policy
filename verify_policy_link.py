# verify_policy_link.py
import os
import sys
import argparse
import numpy as np
import rclpy

# Add current directory to sys.path to find isaac_ros2_env
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print(f"DEBUG: Current Dir: {current_dir}")
print(f"DEBUG: Directory Contents: {os.listdir(current_dir)}")
print(f"DEBUG: sys.path: {sys.path}")

try:
    from isaac_ros2_env import IsaacROS2Env, IsaacROS2Config
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable Baselines 3 not found. Falling back to random actions.")

def check_model_compatibility(model, env):
    """
    Checks if the loaded model's observation and action spaces match the environment.
    """
    print("\n--- Model Compatibility Check ---")
    
    # Check Observation Space
    model_obs_shape = model.observation_space.shape
    env_obs_shape = env.observation_space.shape
    print(f"Model Observation Shape: {model_obs_shape}")
    print(f"Env Observation Shape:   {env_obs_shape}")
    
    if model_obs_shape != env_obs_shape:
        print("WARNING: Observation shape mismatch! Predict() may fail.")
    else:
        print("Observation shapes match.")

    # Check Action Space
    # SB3 action space is often wrapped, so direct comparison can be tricky, 
    # but we can check dimensions for Box spaces.
    try:
        model_act_dim = model.action_space.shape
        env_act_dim = env.action_space.shape
        print(f"Model Action Shape:      {model_act_dim}")
        print(f"Env Action Shape:        {env_act_dim}")
        
        if model_act_dim != env_act_dim:
             print("WARNING: Action shape mismatch!")
        else:
             print("Action shapes match.")
    except AttributeError:
        print("Could not compare action space shapes (likely Discrete vs Box mismatch).")

    print("---------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Verify Isaac Sim <-> ROS2 <-> Policy link.")
    parser.add_argument("--model", type=str, default="final_model_fresh.zip", help="Path to the SB3 policy file (zip).")
    args = parser.parse_args()

    print("Starting Policy Link Verification...")
    
    # resolve absolute path if needed
    policy_path = os.path.abspath(args.model)
    
    # Initialize Environment
    config = IsaacROS2Config()
    env = IsaacROS2Env(config=config)
    
    model = None
    if SB3_AVAILABLE:
        if os.path.exists(policy_path):
            print(f"Loading policy from {policy_path}...")
            try:
                model = PPO.load(policy_path)
                print("Model loaded successfully.")
                check_model_compatibility(model, env)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to random actions.")
        else:
            print(f"WARNING: Model file not found at: {policy_path}")
            print("Running in 'Plumbing Verification Mode' with RANDOM ACTIONS.")
    else:
        print("SB3 library missing. Running with RANDOM ACTIONS.")

    print("\nRunning Verification Loop (100 steps)...")
    try:
        obs, info = env.reset()
        
        for i in range(100):
            if model:
                # Deterministic=True is standard for evaluation
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, done, truncated, info = env.step(action)
            
            if i % 10 == 0:
                print(f"Step {i}: Reward={reward:.4f}, Speed={info.get('speed', 0):.2f}")
                
            if done or truncated:
                print("Episode finished. Resetting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("Verification stopped by user.")
    except Exception as e:
        print(f"Simulation Loop Error: {e}")
    finally:
        env.close()
        print("\nVerification Complete!")

if __name__ == "__main__":
    main()
