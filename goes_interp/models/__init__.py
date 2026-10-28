from goes_interp.models.raft import SatelliteRAFT, get_bidirectional_flow
from goes_interp.models.warp import warp_flow, occlusion_masks
from goes_interp.models.zmap import PhysicsZMapNet
from goes_interp.models.splat import forward_splat, bidirectional_splat
from goes_interp.models.unet import build_unet, make_unet_input, CompositeLoss
from goes_interp.utils import get_device


def build_models(device=None):
    """Instantiate the full pipeline: (raft, zmap, unet) on the given device."""
    device = device or get_device()
    raft = SatelliteRAFT(device=device)
    zmap = PhysicsZMapNet().to(device)
    unet = build_unet(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"U-Net parameters: {n_params:,}")
    return raft, zmap, unet
