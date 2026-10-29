"""Local occupancy predictor F_psi: panoramic RGB -> egocentric local occupancy.

Paper Section IV-B: a convolutional encoder-decoder extended to panoramic
input.  Each of the 4 views goes through the same (weight-shared) encoder; the
per-view features are concatenated in a fixed order (front, left, back,
right), fused by fully connected layers - which is where the network learns
the camera-to-top-down mapping - and decoded to a LOCAL_SIZE x LOCAL_SIZE
grid of occupancy logits (see local_grid.py for the grid convention).
"""

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _conv(cin, cout, k, s):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _deconv(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PanoLocalOccNet(nn.Module):
    def __init__(self, num_views=4, img_hw=(72, 96), out_size=64, view_feat=256, bottleneck=1024):
        super().__init__()
        self.num_views, self.img_hw, self.out_size = num_views, tuple(img_hw), out_size
        self.encoder = nn.Sequential(
            _conv(3, 32, 5, 2), _conv(32, 32, 3, 1),
            _conv(32, 64, 3, 2), _conv(64, 64, 3, 1),
            _conv(64, 128, 3, 2),
            _conv(128, 128, 3, 2),
        )
        with torch.no_grad():
            enc_numel = self.encoder(torch.zeros(1, 3, *img_hw)).numel()
        self.view_fc = nn.Sequential(nn.Flatten(), nn.Linear(enc_numel, view_feat), nn.ReLU(inplace=True))
        self.seed = out_size // 8
        self.fuse = nn.Sequential(
            nn.Linear(num_views * view_feat, bottleneck), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(bottleneck, 128 * self.seed * self.seed), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(_deconv(128, 64), _deconv(64, 32), _deconv(32, 16),
                                     nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, x):
        """x: (B, V, 3, H, W) normalised images -> (B, out, out) occupancy logits."""
        b, v = x.shape[:2]
        f = self.view_fc(self.encoder(x.flatten(0, 1))).view(b, -1)
        z = self.fuse(f).view(b, 128, self.seed, self.seed)
        return self.decoder(z).squeeze(1)


def images_to_tensor(views_rgb_uint8: np.ndarray) -> torch.Tensor:
    """(..., V, H, W, 3) uint8 RGB -> (..., V, 3, H, W) normalised float tensor."""
    x = (views_rgb_uint8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, -3)))


def load_model(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PanoLocalOccNet(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt
