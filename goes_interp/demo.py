"""Pipeline demos: exploratory diagnostics on one triplet and output GIF generation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from goes_interp import config
from goes_interp.data.loaders import load_triplet
from goes_interp.inference import interpolate
from goes_interp.models.raft import get_bidirectional_flow
from goes_interp.models.splat import bidirectional_splat
from goes_interp.models.warp import occlusion_masks
from goes_interp.utils import to_tensor
from goes_interp import visualize


def explore_triplet(triplet, raft_model, zmap_model, device):
    """Run every pipeline stage once on a single triplet and save diagnostic figures.

    Order mirrors the notebook: preview -> RAFT flow -> occlusion masks ->
    physics Z-map -> bidirectional splatting.
    """
    n0, n30, n60 = load_triplet(*triplet)
    visualize.plot_triplet_preview(n0, n30, n60)

    flow_fwd_t, flow_bwd_t = get_bidirectional_flow(raft_model, n0, n60, device=device)
    flow_fwd_np = flow_fwd_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    magnitude = np.linalg.norm(flow_fwd_np, axis=-1)
    print(f"Flow forward : shape={flow_fwd_np.shape} magnitude "
          f"[min={magnitude.min():.3f} max={magnitude.max():.3f} mean={magnitude.mean():.3f}]")
    visualize.plot_flow(n0, flow_fwd_np)

    mask_fwd, mask_bwd = occlusion_masks(flow_fwd_t, flow_bwd_t)
    visualize.plot_masks(mask_fwd, mask_bwd)

    f0_t = to_tensor(n0, device)
    f1_t = to_tensor(n60, device)
    with torch.no_grad():
        z0_t = zmap_model(f0_t)
        z1_t = zmap_model(f1_t)
    visualize.plot_zmap(n0, (1.0 - f0_t).squeeze().cpu().numpy(), z0_t.squeeze().cpu().numpy())

    with torch.no_grad():
        w0, w1, mask_fwd2, mask_bwd2 = bidirectional_splat(
            f0_t, f1_t, flow_fwd_t, flow_bwd_t, z0_t, z1_t, mask_fwd, mask_bwd
        )
    visualize.plot_splat(n0, w0.squeeze().cpu().numpy(),
                         1.0 - mask_fwd2.squeeze().cpu().numpy(),
                         w1.squeeze().cpu().numpy(), n60)
    print("Exploratory pipeline complete.")
    return {"flow": flow_fwd_np, "mask_fwd": mask_fwd, "mask_bwd": mask_bwd,
            "z0": z0_t, "w0": w0, "w1": w1}


def generate_gif(n0, n60, raft_model, zmap_model, unet_model, device, path=None):
    """Animate interpolated frames between the endpoints and save as GIF."""
    frames = []
    for t in config.GIF_TIMES:
        if t == 0.0:
            frames.append(n0)
        elif t == 1.0:
            frames.append(n60)
        else:
            frames.append(interpolate(n0, n60, raft_model, zmap_model, unet_model,
                                      t=t, device=device))

    minute_at = dict(zip(config.GIF_TIMES,
                         (*config.GIF_MINUTE_LABELS, 60)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    image = ax.imshow(frames[0], cmap="inferno", vmin=0, vmax=1)

    def update(index):
        image.set_array(frames[index])
        label = f"Time: T + {minute_at[config.GIF_TIMES[index]]} min"
        if config.GIF_TIMES[index] == 0.5:
            label += "  <- interpolated midpoint"
        ax.set_title(label, fontsize=13, color="white",
                     bbox=dict(facecolor="#333", alpha=0.7, boxstyle="round"))
        return [image]

    animation_obj = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=500, blit=True
    )
    path = path or (config.OUTPUT_DIR / "interpolated_sequence.gif")
    animation_obj.save(str(path), writer="pillow", dpi=100)
    plt.close(fig)
    print(f"Saved GIF: {path}")
    return path
