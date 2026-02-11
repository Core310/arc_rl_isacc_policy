"""
Isaac Sim ROS2 Gym Environment
===
Gymnasium-compatible environment that interfaces with Isaac Sim via ROS2.

Subscriptions:
  - /camera/image_raw (sensor_msgs/Image): 128x128 RGB camera
  - /vehicle_state (custom or nav_msgs/Odometry): vehicle telemetry

Publications:
  - /ackermann_cmd (ackermann_msgs/AckermannDrive): control commands

Observation Space:
  - image: (128, 128, 3) uint8
  - vec: (12, ) float32 telemetry vector

Action Space:
  - (3,) continuous: [steer, throttle, brake]
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from threading import Lock, Event
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# ROS2 message imports
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Import lane detector for visual rewards
from lane_detector import SimpleLaneDetector, LaneDetectionResult


