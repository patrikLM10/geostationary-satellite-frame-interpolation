"""U-Net synthesis network and composite training loss.

The U-Net predicts a *residual* correction on top of the splatted blend:
    pred = clamp(0.5*w0 + 0.5*w1 + (unet_out - 0.5), 0, 1)
so an untrained network cannot regress below the geometric baseline.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import piq

from goes_interp import config
from goes_interp.utils import get_device

L1_WEIGHT = config.L1_WEIGHT
SSIM_WEIGHT = config.SSIM_WEIGHT


def build_unet(device=None):
    device = device or get_device()
    try:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=8,
            classes=1,
            activation="sigmoid",
        )
        print("U-Net: ResNet34 encoder with ImageNet weights.")
    except Exception as exc:  # noqa: BLE001
        print(f"ImageNet weights failed ({exc}). Initializing from scratch.")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=8,
            classes=1,
            activation="sigmoid",
        )
    return model.to(device)


def make_unet_input(w0, w1, flow_fwd, flow_bwd, mask_fwd, mask_bwd):
    """8-channel input: 2 warped frames + 2 normalized flows + 2 masks."""
    _, _, height, width = w0.shape
    flow_fwd_n = flow_fwd / max(height, width)
    flow_bwd_n = flow_bwd / max(height, width)
    return torch.cat([w0, w1, flow_fwd_n, flow_bwd_n, mask_fwd, mask_bwd], dim=1)


class CompositeLoss(nn.Module):
    """Weighted L1 + (1 - SSIM): L1_WEIGHT * L1 + SSIM_WEIGHT * SSIM-loss."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = 1.0 - piq.ssim(pred, target, data_range=1.0)
        return L1_WEIGHT * loss_l1 + SSIM_WEIGHT * loss_ssim
