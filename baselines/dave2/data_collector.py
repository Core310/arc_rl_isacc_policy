import csv
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import cv2

from .experts import ScriptedExpert

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collects expert driving data and saves to disk.

    This is a pure data recorder - it takes numpy arrays (image + telemetry + action)
    and write PNG frames + CSV labels. It has zero knowledge of where the data comes from
    (ROS2, direct API, replay, mock env, etc.).

    The expert controller is optional - if provided, the collector computes the expert action from telemetry.
    If not, the caller passes the actio directly via collect_from_arrays_with_action().
    """

    def __init__(
        self,
        output_dir: str,
        collection_hz: int = 10,
        expert: Optional[ScriptedExpert] = None,
        img_width: int = 160,
        img_height: int = 90,
    ):
        """
        Args:
            output_dir: Directory to save collected data.
            collection_hz: Rate at which to save frames (Hz).
            expert: Expert controller instance. Defaults to ScriptedExpert.
                Only used by collect_from_arrays() - collect_from_arrays_with_action()
                ignores this and uses the caller-provided action.
            img_width: Expected camera image width (for metadata only).
            img_height: Expected camera image height (for metadata only).
        """
        self.output_dir = Path(output_dir)
        self.collection_hz = collection_hz
        self.expert = expert or ScriptedExpert()
        self.img_width = img_width
        self.img_height = img_height

        # Create output directory structure
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._frame_count = 0
        self._csv_writer = None
        self._csv_file = None

    def collect_from_arrays(
        self,
        image: np.ndarray,
        telemetry: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Record a frame using the internal expert to compute the action.

        This is used by scripted collection where the expert controller determines the action from telemetry.

        Args:
            image: (H, W, 3) RGB uint8 camera image.
            telemetry: (12,) float32 telemetry vector.

        Returns:
            Expert action [steering, throttle, brake].
        """
        action = self.expert.compute_action(telemetry)
        self._save_frame(image, action, telemetry)
        return action

    def collect_from_arrays_with_action(
        self,
        image: np.ndarray,
        telemetry: np.ndarray,
        action: np.ndarray,
    ):
        """
        Record a frame with a caller-provided action.

        This is used by teleop collection where the human provides the action and we just record
        what they did.

        Args:
            image: (H, W, 3) RGB uint8 camera image.
            telemetry: (12,) float32 telemetry vector.
            action: (3,) float32 [steering, throttle, brake] from the human.
        """
        self._save_frame(image, action, telemetry)

    def _save_frame(
        self,
        image: np.ndarray,
        action: np.ndarray,
        telemetry: np.ndarray,
    ):
        """Internal: write one image + label row to disk."""
        frame_id = f"frame_{self._frame_count:06d}"

        # Save image as PNG (lossless), RGB -> BGR for OpenCV
        frame_path = self.frames_dir / f"{frame_id}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Write label row
        if self._csv_writer is None:
            self._csv_file = open(self.output_dir / "labels.csv", "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame_id", "steering", "throttle", "brake", "speed"
            ])

        self._csv_writer.writerow([
            frame_id,
            f"{action[0]:.6f}",
            f"{action[1]:.6f}",
            f"{action[2]:.6f}",
            f"{telemetry[3]:.6f}", # speed
        ])

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.info(f"Collected {self._frame_count} frames")

    def save_metadata(self, duration: float = 0.0, expert_name: str = ""):
        """Save collection metadata as YAML."""
        import yaml

        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_frames": self._frame_count,
            "collection_hz": self.collection_hz,
            "duration_seconds": duration,
            "image_resolution": [self.img_width, self.img_height],
            "expert_type": expert_name or self.expert.__class__.__name__,
            "expert_params": {
                "kp_steer": getattr(self.expert, "kp_steer", None),
                "kd_steer": getattr(self.expert, "kd_steer", None),
                "target_speed": getattr(self.expert, "target_speed", None),
            },
        }

        with open(self.output_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(
            f"Collection complete: {self._frame_count} frames "
            f"saved to {self.output_dir}"
        )

    def close(self):
        """Flush and close output files."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
