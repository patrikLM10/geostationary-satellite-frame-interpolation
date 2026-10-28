"""Inference: fold endpoint frames into intermediate time t in (0, 1)."""

import torch

from goes_interp import config
from goes_interp.models.splat import bidirectional_splat
from goes_interp.models.unet import make_unet_input
from goes_interp.models.warp import occlusion_masks
from goes_interp.utils import get_device, to_tensor


def load_checkpoints(zmap_model, unet_model, device, checkpoint_dir=None):
    """Load best checkpoints if present. Returns True when loaded."""
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    zmap_path = checkpoint_dir / config.ZMAP_CKPT
    unet_path = checkpoint_dir / config.UNET_CKPT
    if not (zmap_path.exists() and unet_path.exists()):
        print("Warning: checkpoints not found; using current weights.")
        return False
    zmap_model.load_state_dict(torch.load(zmap_path, map_location=device))
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    zmap_model.eval()
    unet_model.eval()
    print("Best checkpoints loaded.")
    return True


def interpolate(norm0, norm60, raft, zmap, unet, t=0.5, device=None):
    """Predict frame at fractional position t between the two endpoint frames."""
    device = device or get_device()
    f0 = to_tensor(norm0, device)
    f1 = to_tensor(norm60, device)

    with torch.no_grad():
        flow_fwd = raft(f0, f1) * t
        flow_bwd = raft(f1, f0) * (1 - t)
        mask_fwd, mask_bwd = occlusion_masks(flow_fwd, flow_bwd)

        z0 = zmap(f0)
        z1 = zmap(f1)

        w0, w1, mask_fwd2, mask_bwd2 = bidirectional_splat(
            f0, f1, flow_fwd, flow_bwd, z0, z1, mask_fwd, mask_bwd
        )

        unet_out = unet(make_unet_input(w0, w1, flow_fwd, flow_bwd, mask_fwd2, mask_bwd2))
        base_blend = 0.5 * w0 + 0.5 * w1
        pred = torch.clamp(base_blend + (unet_out - 0.5), 0.0, 1.0)

    return pred.squeeze().cpu().numpy()
