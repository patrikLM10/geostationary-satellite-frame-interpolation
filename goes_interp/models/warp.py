"""Backward warping with `grid_sample` and occlusion detection.

Occlusion masks come from a forward-backward consistency check: a pixel
is considered trustworthy when forward + warped-backward flow agree.
"""

import torch
import torch.nn.functional as F


def warp_flow(frame_tensor, flow_tensor):
    """Backward-warp `frame_tensor` by sampling it at shifted coordinates."""
    batch, _, height, width = frame_tensor.size()
    yy, xx = torch.meshgrid(
        torch.arange(height, device=frame_tensor.device),
        torch.arange(width, device=frame_tensor.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).repeat(batch, 1, 1, 1)
    vgrid = grid + flow_tensor
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(width - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(height - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(
        frame_tensor, vgrid,
        align_corners=True, mode="bilinear", padding_mode="border",
    )


def occlusion_masks(flow_fwd_t, flow_bwd_t, threshold=1.0):
    """Return (mask_fwd, mask_bwd): 1 = occlusion-consistent, 0 = hole."""
    err_fwd = torch.norm(flow_fwd_t + warp_flow(flow_bwd_t, flow_fwd_t), dim=1, keepdim=True)
    mask_fwd = (err_fwd < threshold).float()

    err_bwd = torch.norm(flow_bwd_t + warp_flow(flow_fwd_t, flow_bwd_t), dim=1, keepdim=True)
    mask_bwd = (err_bwd < threshold).float()
    return mask_fwd, mask_bwd
