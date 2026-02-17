
import sys
import os

print("Current Working Directory:", os.getcwd())
print("
System Path:")
for p in sys.path:
    print(p)

print("
Attempting to import isaac_ros2_env...")
try:
    import isaac_ros2_env
    print("SUCCESS: Imported isaac_ros2_env")
    print("File location:", isaac_ros2_env.__file__)
except ImportError as e:
    print(f"FAILURE: {e}")

print("
Checking for file existence:")
target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_ros2_env.py")
print(f"Looking for: {target_file}")
print(f"Exists: {os.path.exists(target_file)}")
