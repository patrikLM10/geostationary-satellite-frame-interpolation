"""All figure/report generation (saved as PNGs under outputs/)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from goes_interp import config


def save_fig(fig, name):
    path = config.OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path.name}")
    return path


def plot_triplet_preview(n0, n30, n60):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, frame, title in zip(
        axes, (n0, n30, n60),
        ("I0 (0 min)", "I30 (30 min - GT)", "I60 (60 min)"),
    ):
        ax.imshow(frame, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    return save_fig(fig, "01_triplet_preview.png")


def plot_flow(norm0, flow_fwd_np):
    hsv = np.zeros((*flow_fwd_np.shape[:2], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_fwd_np[..., 0], flow_fwd_np[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[0].set_title("Color wheel (hue = direction, sat = magnitude)")
    axes[0].axis("off")

    step = 16
    yy, xx = np.mgrid[0:norm0.shape[0]:step, 0:norm0.shape[1]:step]
    axes[1].imshow(norm0, cmap="gray")
    axes[1].quiver(xx, yy, flow_fwd_np[0::step, 0::step, 0], flow_fwd_np[0::step, 0::step, 1],
                   color="red", angles="xy", scale_units="xy", scale=1)
    axes[1].set_title("Quiver: motion vectors on I0")
    axes[1].axis("off")
    return save_fig(fig, "02_optical_flow.png")


def plot_masks(mask_fwd, mask_bwd):
    mf = mask_fwd.squeeze().cpu().numpy()
    mb = mask_bwd.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mf, cmap="gray")
    axes[0].set_title(f"Mask FWD (valid = white) | holes: {100 * (1 - mf.mean()):.2f}%")
    axes[1].imshow(1.0 - mf, cmap="hot")
    axes[1].set_title("Holes (occluded/disoccluded)")
    axes[2].imshow(mb, cmap="gray")
    axes[2].set_title(f"Mask BWD (valid = white) | holes: {100 * (1 - mb.mean()):.2f}%")
    for ax in axes:
        ax.axis("off")
    return save_fig(fig, "03_occlusion_masks.png")


def plot_zmap(frame_np, z_prior_np, z_out_np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame_np, cmap="gray")
    axes[0].set_title("Input frame")
    im1 = axes[1].imshow(z_prior_np, cmap="inferno")
    axes[1].set_title("Z prior (1 - frame)")
    axes[2].imshow(z_out_np, cmap="inferno")
    axes[2].set_title("Z map output")
    fig.colorbar(im1, ax=axes[1:], orientation="horizontal", fraction=0.05, pad=0.05)
    for ax in axes:
        ax.axis("off")
    return save_fig(fig, "04_physics_zmap.png")


def plot_splat(norm0, w0_np, holes0, w1_np, n60):
    w0_color = plt.cm.gray(w0_np)[:, :, :3]
    w0_color[holes0 > 0.5] = [1, 0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, frame, title in zip(
        axes, (norm0, w0_color, w1_np, n60),
        ("I0 (0 min)", "warped0 -> t=0.5 (holes = red)", "warped1 -> t=0.5", "I60 (60 min)"),
    ):
        ax.imshow(frame, cmap="gray" if frame.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    return save_fig(fig, "05_softmax_splatting.png")


def plot_loss_curve(history, epochs):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, epochs + 1), history["loss"], marker="o", markersize=3, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Composite loss (0.2 L1 + 0.8 SSIM)")
    ax.set_title("Training loss curve (30-min triplets)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_fig(fig, "06_training_loss_curve.png")


def plot_comparison(n0, linear, pred, gt, n60, ssim_lin=None, ssim_pred=None):
    def title(name, ssim):
        return f"{name}\nSSIM: {ssim:.4f}" if ssim is not None else name

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    panels = (
        (n0, "I0 (0 min)", None),
        (linear, title("Linear interp", ssim_lin), None),
        (pred, title("Model pred", ssim_pred), None),
        (gt, "Ground truth (30 min)", None),
        (n60, "I60 (60 min)", None),
    )
    for ax, (frame, title_text, _) in zip(axes, panels):
        ax.imshow(frame, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title_text)
        ax.axis("off")
    return save_fig(fig, "07_interpolation_comparison.png")


def plot_benchmark(results):
    summary = results["summary"]
    best, worst = results["best"], results["worst"]

    fig = plt.figure(figsize=(24, 10))
    labels = ["SSIM", "PSNR (dB)", "MSE (x10-3)", "FSIM"]
    pairs = (("ssim", "higher is better"), ("psnr", "higher is better"),
             ("mse", "lower is better"), ("fsim", "higher is better"))
    for i, (metric, hint) in enumerate(pairs):
        lv = summary[f"lin_{metric}"]["mean"]
        mv = summary[f"mod_{metric}"]["mean"]
        ax = fig.add_subplot(2, 4, i + 1)
        bars = ax.bar(["Linear", "Model"], [lv, mv], color=["#4878CF", "#6ACC65"],
                      width=0.5, edgecolor="white")
        ax.set_title(f"{labels[i]} ({hint})", fontsize=12, fontweight="bold")
        ax.set_ylim(min(lv, mv) * 0.9, max(lv, mv) * 1.1)
        for bar, val in zip(bars, (lv, mv)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        delta = mv - lv
        good = delta > 0 if metric != "mse" else delta < 0
        sign = "+" if delta >= 0 else ""
        color = "green" if good else "red"
        arrow = "^" if good else "v"
        ax.set_xlabel(f"Delta = {sign}{delta:.3f}  {arrow}", color=color, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    vmax = max(
        np.abs(best["pred"] - best["gt"]).max(),
        np.abs(worst["pred"] - worst["gt"]).max(),
    )
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(best["pred"], cmap="inferno", vmin=0, vmax=1)
    ax5.set_title(f"Best - model pred (SSIM {best['ssim']:.3f})")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(best["pred"] - best["gt"], cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax6.set_title("Best - pred vs GT diff")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(worst["pred"], cmap="inferno", vmin=0, vmax=1)
    ax7.set_title(f"Worst - model pred (SSIM {worst['ssim']:.3f})")
    ax7.axis("off")

    crop = (slice(80, 176), slice(80, 176))
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.imshow(worst["linear"][crop], cmap="inferno", vmin=0, vmax=1)
    ax8.set_title("Worst - linear zoom (96x96)")
    ax8.axis("off")

    fig.suptitle(
        "Benchmark: RAFT + softmax splatting + U-Net  vs  linear interpolation\n"
        "GOES-19 Channel 13, 30-min gaps", fontsize=13, fontweight="bold", y=1.02,
    )
    return save_fig(fig, "08_benchmark.png")


def _summary_value(summary, side, metric):
    entry = summary[f"{side}_{metric}"]
    return entry["mean"]


def summary_box(results, n_train, n_val):
    summary = results["summary"]
    best_psnr = max(results["metrics"]["mod_psnr"])
    lines = [
        "=" * 60,
        " SATELLITE FRAME INTERPOLATION - RESULTS",
        "=" * 60,
        f" Model           : RAFT + softmax splatting + U-Net",
        f" Training data   : GOES-19 ABI Channel 13 (self-supervised)",
        f" Triplet gap     : 30 min (I0, I30, I60)",
        f" Train triplets  : {n_train}",
        f" Val triplets    : {n_val}",
        f" Best val SSIM   : {results['best']['ssim']:.3f}",
        f" Best val PSNR   : {best_psnr:.1f} dB",
        f" Linear SSIM     : {_summary_value(summary, 'lin', 'ssim'):.3f} "
        f"(delta {_summary_value(summary, 'mod', 'ssim') - _summary_value(summary, 'lin', 'ssim'):+.3f})",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print()
    print(text)
    return text
