"""Bidirectional z-weighted softmax splatting.

When the sniklaus CUDA kernel is built (see goes_interp.setup_splatting),
forward splatting uses it directly; otherwise a backward-warp approximation
via `grid_sample` is used.
"""

import torch

from goes_interp import setup_splatting
from goes_interp.models.warp import warp_flow


def forward_splat(frame_t, flow_t, z_t):
    if setup_splatting.SPLAT_CUDA:
        return setup_splatting.softsplat_splat(frame_t, flow_t, z_t)
    return warp_flow(frame_t, -flow_t)


def bidirectional_splat(f0, f1, flow_fwd, flow_bwd, z0, z1, mask_fwd, mask_bwd):
    """Splat both endpoint frames to the midpoint, plus their occlusion masks."""
    warped0 = forward_splat(f0, flow_fwd, z0)
    warped1 = forward_splat(f1, flow_bwd, z1)
    return warped0, warped1, mask_fwd, mask_bwd
