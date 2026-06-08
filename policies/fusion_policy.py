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
        
        # Mastery Fix: Partial-Brain unfreeze (Only Layer 4 + Pooling/FC)
        # This provides track-specific precision while fitting in 12GB VRAM.
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        # Remove the final fully connected layer (identity passthrough)
        # ResNet18 output dim is 512
        self.resnet.fc = nn.Identity()

        # ImageNet Normalization Constants (Register as buffers for GPU efficiency)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 2. Physics Stream
        vec_dim = observation_space["vec"].shape[0]

        # 3. Fusion Head
        # ResNet-18 features (512) + Telemetry (vec_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + vec_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # Mastery Fix: Explicit Kaiming (He) initialization for fusion layers
        for m in self.fusion_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observation: dict) -> torch.Tensor:
        # 1. ImageNet Normalization (On GPU)
        img = observation["image"].float()
        
        # Mastery Fix: The environment already normalizes to [0, 1]. 
        # We only apply ImageNet mean/std if the image is raw uint8 [0, 255].
        # If it's already [0, 1], we pass it raw to the ResNet to avoid 'Black Road' syndrome.
        if img.max() > 1.0:
            img = img / 255.0
            img = (img - self.mean) / self.std
        
        # 2. Extract ResNet features (B, 512)
        visual_feats = self.resnet(img)

        # 3. Fusion (Vision + Telemetry)
        combined = torch.cat([visual_feats, observation["vec"]], dim=1)

        return self.fusion_head(combined)
