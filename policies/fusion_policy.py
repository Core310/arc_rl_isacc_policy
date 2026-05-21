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
from torchvision.models import resnet18, ResNet18_Weights

class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """
    HD Fusion Network with ResNet-18 Backbone.
    Uses ImageNet-v1 weights (unfrozen) for rapid feature convergence.
    Handles uint8 images with on-GPU normalization.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Pass the final output dim (256) to SB3
        super().__init__(observation_space, features_dim=features_dim)

        # 1. Vision Stream: ResNet-18
        # We use weights=ResNet18_Weights.IMAGENET1K_V1 for transfer learning
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer (identity passthrough)
        # ResNet18 output dim is 512
        self.resnet.fc = nn.Identity()

        # 2. Physics Stream
        vec_dim = observation_space["vec"].shape[0]

        # 3. Fusion Head
        # ResNet-18 features (512) + Telemetry (vec_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + vec_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observation: dict) -> torch.Tensor:
        # 1. Normalize uint8 images to [0, 1] on GPU
        img = observation["image"].float() / 255.0
        
        # 2. Extract ResNet features (B, 512)
        visual_feats = self.resnet(img)

        # 3. Fusion (Vision + Telemetry)
        combined = torch.cat([visual_feats, observation["vec"]], dim=1)

        return self.fusion_head(combined)
