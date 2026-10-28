"""Optical flow between satellite frames.

Primary: torchvision RAFT-small (3-channel adapted grayscale input).
Fallback: OpenCV Farneback (CPU) when weights can't be downloaded.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from goes_interp.utils import get_device, to_tensor


class SatelliteRAFT(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or get_device()
        self.model = None
        self.transforms = None
        self.use_raft = False
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False).to(self.device).eval()
            self.transforms = weights.transforms()
            self.use_raft = True
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: RAFT failed to load ({exc}). Using Farneback fallback.")

    def forward(self, f0, f1):
        batch, _, height, width = f0.shape

        if not self.use_raft:
            flows = []
            for i in range(batch):
                img0 = (f0[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                img1 = (f1[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flows.append(torch.from_numpy(flow).permute(2, 0, 1))
            return torch.stack(flows).to(f0.device)

        f0_rgb = f0.repeat(1, 3, 1, 1) * 255.0
        f1_rgb = f1.repeat(1, 3, 1, 1) * 255.0

        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h > 0 or pad_w > 0:
            f0_rgb = torch.nn.functional.pad(f0_rgb, (0, pad_w, 0, pad_h))
            f1_rgb = torch.nn.functional.pad(f1_rgb, (0, pad_w, 0, pad_h))

        f0_t, f1_t = self.transforms(f0_rgb, f1_rgb)
        with torch.no_grad():
            flow = self.model(f0_t, f1_t)[-1]
        return flow[:, :, :height, :width]


def get_bidirectional_flow(raft_model, frame0_np, frame60_np, t=0.5, device=None):
    """Scaled forward I0->I60 and backward I60->I0 flow at interpolation time t."""
    device = device or get_device()
    t0 = to_tensor(frame0_np, device)
    t1 = to_tensor(frame60_np, device)
    with torch.no_grad():
        flow_fwd = raft_model(t0, t1)
        flow_bwd = raft_model(t1, t0)
    return flow_fwd * t, flow_bwd * (1 - t)
