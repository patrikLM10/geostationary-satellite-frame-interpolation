"""Self-supervised training loop over GOES-19 triplets.

RAFT is frozen (inference mode); the Z-map and U-Net are trained jointly
with AMP, gradient clipping and a cosine-annealed LR schedule.
"""

import numpy as np
import piq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from goes_interp import config
from goes_interp.data.dataset import GOESTripletDataset
from goes_interp.models.splat import bidirectional_splat
from goes_interp.models.unet import CompositeLoss, make_unet_input
from goes_interp.models.warp import occlusion_masks


def _autocast(device, enabled):
    try:
        return torch.amp.autocast(device_type=device, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler(enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(raft, zmap, unet, loader, optimizer, scaler, device):
    unet.train()
    zmap.train()
    raft.eval()

    criterion = CompositeLoss()
    epoch_loss = 0.0
    epoch_ssim = 0.0
    use_amp = device == "cuda"

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        f0 = batch["frame0"].to(device)
        f1 = batch["frame1"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            flow_fwd = raft(f0, f1) * 0.5
            flow_bwd = raft(f1, f0) * 0.5
            mask_fwd, mask_bwd = occlusion_masks(flow_fwd, flow_bwd)

        with _autocast(device, use_amp):
            z0 = zmap(f0)
            z1 = zmap(f1)
            w0, w1, mask_fwd2, mask_bwd2 = bidirectional_splat(
                f0, f1, flow_fwd, flow_bwd, z0, z1, mask_fwd, mask_bwd
            )
            unet_out = unet(make_unet_input(w0, w1, flow_fwd, flow_bwd, mask_fwd2, mask_bwd2))
            base_blend = 0.5 * w0 + 0.5 * w1
            pred = torch.clamp(base_blend + (unet_out - 0.5), 0.0, 1.0)
            loss = criterion(pred, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(zmap.parameters()), config.GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(zmap.parameters()), config.GRAD_CLIP
            )
            optimizer.step()

        epoch_loss += loss.item()
        with torch.no_grad():
            ssim_val = piq.ssim(pred.float(), target.float(), data_range=1.0).item()
        epoch_ssim += ssim_val
        progress.set_postfix({"loss": f"{loss.item():.4f}", "ssim": f"{ssim_val:.4f}"})

    n_batches = max(1, len(loader))
    return epoch_loss / n_batches, epoch_ssim / n_batches


def train(raft_model, zmap_model, unet_model, train_triplets, device,
          epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
          checkpoint_dir=None):
    """Run the full training loop, save checkpoints and the loss curve. Returns history."""
    if not train_triplets:
        print("No training triplets available. Skipping training.")
        return {"loss": [], "ssim": []}

    dataset = GOESTripletDataset(train_triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=config.NUM_WORKERS)

    params = list(unet_model.parameters()) + list(zmap_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = _grad_scaler(device == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": [], "ssim": []}
    print(f"Training {epochs} epochs on {len(train_triplets)} triplets "
          f"(30-min gaps, self-supervised)...")

    for epoch in range(epochs):
        loss, ssim = train_one_epoch(
            raft_model, zmap_model, unet_model, loader, optimizer, scaler, device
        )
        scheduler.step()
        history["loss"].append(loss)
        history["ssim"].append(ssim)
        print(f"Epoch {epoch + 1:02d}/{epochs} | loss: {loss:.4f} | SSIM: {ssim:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(zmap_model.state_dict(), checkpoint_dir / config.ZMAP_CKPT)
    torch.save(unet_model.state_dict(), checkpoint_dir / config.UNET_CKPT)
    print(f"Checkpoints saved to {checkpoint_dir}")

    from goes_interp import visualize

    visualize.plot_loss_curve(history, epochs)
    return history
