"""
models/occupancy_model.py
--------------------------
ResNet18 encoder + transpose-convolution decoder for occupancy grid prediction.

Architecture
------------
Encoder : ResNet18 pretrained on ImageNet, classification head removed.
          Feature map progression (input 224×224):
              conv1+pool  →  (B,  64,  56, 56)
              layer1      →  (B,  64,  56, 56)
              layer2      →  (B, 128,  28, 28)
              layer3      →  (B, 256,  14, 14)
              layer4      →  (B, 512,   7,  7)   ← bottleneck

Decoder : Five upsampling stages via ConvTranspose2d.
          Each stage (except final): ConvTranspose2d → BN → ReLU.
          Final stage: Conv2d → 3 logits (no activation).

          7  → 14  → 28  → 56  → 112 → 64
          (last step uses adaptive_avg_pool to hit exact 64×64)

Output  : (B, 3, 64, 64) raw logits for {unknown, occupied, free}.
          Directly compatible with nn.CrossEntropyLoss.

Input channel adaptation
------------------------
in_channels == 3 : standard pretrained conv1, nothing changed.
in_channels  > 3 : new conv1 created; pretrained RGB weights copied into
                   channels [:3]; extra channels initialised as the channel-
                   mean of the RGB weights (better than random, avoids bias
                   towards one colour channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet18_Weights, resnet18


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    All architectural hyper-parameters in one place.
    Keeping config separate from the model class allows serialisation to YAML
    and makes reproducing experiments trivial.
    """
    in_channels:    int   = 3       # 3 | 4 | 12 | 16
    num_classes:    int   = 3       # unknown / occupied / free
    grid_size:      int   = 64      # output H == W
    pretrained:     bool  = True    # load ImageNet weights for encoder
    freeze_encoder: bool  = False   # freeze encoder during early training


# ─────────────────────────────────────────────────────────────────────────────
# Decoder building block
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single upsampling stage:
        ConvTranspose2d(in_ch, out_ch, kernel, stride, padding)
        → BatchNorm2d
        → ReLU(inplace=True)

    Parameters chosen so that spatial resolution doubles exactly.
    kernel=4, stride=2, padding=1  →  H_out = 2 * H_in  (for integer H_in).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18Encoder(nn.Module):
    """
    ResNet18 backbone with classification head removed.
    Returns the bottleneck feature map: (B, 512, H/32, W/32).

    For in_channels != 3 the first convolution is replaced and pretrained
    RGB weights are retained in the first 3 channels.
    """

    BOTTLENECK_CHANNELS: int = 512

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # ── Adapt first conv if needed ────────────────────────────────────
        if in_channels != 3:
            backbone.conv1 = self._adapt_conv1(backbone.conv1, in_channels)

        # ── Strip classification head ─────────────────────────────────────
        # Keep: conv1, bn1, relu, maxpool, layer1–4
        # Drop: avgpool, fc
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4

    @staticmethod
    def _adapt_conv1(original: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        """
        Replace conv1 for in_channels != 3.

        Weight initialisation strategy:
            channels  0:3  ← pretrained RGB weights  (exact copy)
            channels  3:   ← mean of RGB weight channels (per filter)

        This is strictly better than:
            - random init  (throws away pretrained knowledge entirely)
            - duplicating R (introduces colour-channel bias)
        """
        out_ch, _, kH, kW = original.weight.shape
        new_conv = nn.Conv2d(
            in_channels, out_ch,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original.weight          # RGB
            rgb_mean = original.weight.mean(dim=1, keepdim=True)     # (out,1,k,k)
            for c in range(3, in_channels):
                new_conv.weight[:, c:c+1, :, :] = rgb_mean

        return new_conv

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)          # (B,  64, 56, 56)  for 224×224 input
        x = self.layer1(x)           # (B,  64, 56, 56)
        x = self.layer2(x)           # (B, 128, 28, 28)
        x = self.layer3(x)           # (B, 256, 14, 14)
        x = self.layer4(x)           # (B, 512,  7,  7)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyDecoder(nn.Module):
    """
    Transpose-convolution decoder.

    Upsampling path (bottleneck → output):
        (B, 512,  7,  7)
        (B, 256, 14, 14)  ← DecoderBlock
        (B, 128, 28, 28)  ← DecoderBlock
        (B,  64, 56, 56)  ← DecoderBlock
        (B,  32,112,112)  ← DecoderBlock
        AdaptiveAvgPool2d → (B, 32, 64, 64)
        Conv2d            → (B,  3, 64, 64)  [logits]

    AdaptiveAvgPool is used for the 112→64 step (non-power-of-2 target)
    to avoid the checkerboard artefacts that TransposedConv with non-integer
    scale factors can introduce.
    """

    def __init__(self, num_classes: int = 3, grid_size: int = 64) -> None:
        super().__init__()
        self.grid_size = grid_size

        self.up1 = DecoderBlock(512, 256)   #  7 → 14
        self.up2 = DecoderBlock(256, 128)   # 14 → 28
        self.up3 = DecoderBlock(128,  64)   # 28 → 56
        self.up4 = DecoderBlock( 64,  32)   # 56 → 112

        # 112 → grid_size (64): adaptive pool avoids checkerboard artefacts
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Final projection: no BN, no activation → raw logits for CrossEntropy
        self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up1(x)   # (B, 256, 14, 14)
        x = self.up2(x)   # (B, 128, 28, 28)
        x = self.up3(x)   # (B,  64, 56, 56)
        x = self.up4(x)   # (B,  32,112,112)
        x = self.pool(x)  # (B,  32, 64, 64)
        x = self.head(x)  # (B,   3, 64, 64)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyModel(nn.Module):
    """
    ResNet18 encoder + transpose-conv decoder for occupancy grid prediction.

    Parameters
    ----------
    cfg : ModelConfig
        All architectural hyper-parameters.

    Forward
    -------
    input  : (B, C, H, W)  float32   C = cfg.in_channels, H=W=224 recommended
    output : (B, 3, 64, 64) float32  raw logits

    Usage
    -----
    >>> cfg   = ModelConfig(in_channels=3)
    >>> model = OccupancyModel(cfg)
    >>> logits = model(torch.randn(2, 3, 224, 224))
    >>> logits.shape
    torch.Size([2, 3, 64, 64])
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg     = cfg or ModelConfig()
        self.encoder = ResNet18Encoder(
            in_channels = self.cfg.in_channels,
            pretrained  = self.cfg.pretrained,
        )
        self.decoder = OccupancyDecoder(
            num_classes = self.cfg.num_classes,
            grid_size   = self.cfg.grid_size,
        )

        if self.cfg.freeze_encoder:
            self._freeze_encoder()

    # ── public ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)   # (B, 512, 7, 7)
        logits   = self.decoder(features)  # (B, 3, 64, 64)
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        params = (
            self.parameters() if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(math.prod(p.shape) for p in params)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for fine-tuning after decoder warm-up."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self) -> None:
        self._freeze_encoder()

    # ── private ──────────────────────────────────────────────────────────────

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def __repr__(self) -> str:
        total     = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return (
            f"OccupancyModel(\n"
            f"  in_channels={self.cfg.in_channels}, "
            f"num_classes={self.cfg.num_classes}, "
            f"grid_size={self.cfg.grid_size}\n"
            f"  total params:     {total:,}\n"
            f"  trainable params: {trainable:,}\n"
            f")"
        )
