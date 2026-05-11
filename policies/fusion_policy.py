"""
Fusion Features Extractor for HD Multi-Modal Perception (VRAM Efficient)

This module implements a flexible CNN vision backbone fused with physics telemetry.
Optimized for 480x270 or 960x540 resolution using early pooling to maintain VRAM stability.

Architecture:
    Visual Stream: CNN (NatureCNN-style with Early Pooling) -> 256-dim features
    Physics Stream: Identity passthrough -> 12-dim telemetry
    Fusion: Vision-Dominant Concatenation + LayerNorm -> 256-dim output

Author: Gemini CLI / Aaron Hamil
Date: 2026-05-10
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Flexible High-Resolution Fusion Network.
    Uses early downsampling to process HD images without exploding connection counts.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Pass the final output dim (256) to SB3
        super().__init__(observation_space, features_dim=features_dim)

        # 1. Image Geometry
        c, h, w = observation_space["image"].shape
        
        # 2. Vision Stream (NatureCNN style with automatic dimensionality)
        # We add an adaptive pooling layer at the start to ensure the CNN 
        # always sees a consistent resolution regardless of the input.
        self.cnn = nn.Sequential(
            # Force HD input into a manageable internal representation
            nn.AdaptiveAvgPool2d((128, 128)),
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 3. Compute CNN output dimension dynamically
        with torch.no_grad():
            sample = torch.as_tensor(observation_space["image"].sample()[None]).float()
            # Note: SB3 handles normalization but we just need the shape here
            n_flatten = self.cnn(sample).shape[1]

        self.vision_linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU()
        )

        # 4. Fusion Head
        vec_dim = observation_space["vec"].shape[0]
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + vec_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observation: dict) -> torch.Tensor:
        # Extract visual features (B, 256)
        visual_feats = self.cnn(observation["image"])
        visual_feats = self.vision_linear(visual_feats)

        # Fusion
        combined = torch.cat([visual_feats, observation["vec"]], dim=1)

        return self.fusion_head(combined)
