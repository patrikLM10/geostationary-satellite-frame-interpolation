"""Benchmark: model interpolation vs linear interpolation on validation triplets."""

import numpy as np
import piq
import torch
from tqdm import tqdm

from goes_interp import config
from goes_interp.data.loaders import load_triplet
from goes_interp.inference import interpolate

METRIC_PAIRS = ("ssim", "psnr", "mse", "fsim")
METRIC_KEYS = tuple(f"{side}_{m}" for side in ("lin", "mod") for m in METRIC_PAIRS)


def _score(pred_t, target_t):
    def ssim(a, b):
        return piq.ssim(a, b, data_range=1.0).item()

    def psnr(a, b):
        return piq.psnr(a, b, data_range=1.0).item()

    def mse(a, b):
        return torch.nn.functional.mse_loss(a, b).item()

    def fsim(a, b):
        try:
            return piq.fsim(a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1),
                            data_range=1.0, chromatic=False).item()
        except Exception:  # noqa: BLE001
            return 0.0

    return {"ssim": ssim(pred_t, target_t), "psnr": psnr(pred_t, target_t),
            "mse": mse(pred_t, target_t), "fsim": fsim(pred_t, target_t)}


def evaluate(val_triplets, raft_model, zmap_model, unet_model, device):
    """Evaluate all validation triplets. Returns a results dict with metrics and extremes."""
    metrics = {key: [] for key in METRIC_KEYS}
    best = {"ssim": -1.0}
    worst = {"ssim": 2.0}

    for triplet in tqdm(val_triplets, desc="Benchmarking"):
        n0, n30, n60 = load_triplet(*triplet)

        pred_mod = interpolate(n0, n60, raft_model, zmap_model, unet_model, t=0.5, device=device)
        pred_lin = 0.5 * n0 + 0.5 * n60

        target_t = torch.from_numpy(n30).unsqueeze(0).unsqueeze(0)
        mod_scores = _score(torch.from_numpy(pred_mod).unsqueeze(0).unsqueeze(0), target_t)
        lin_scores = _score(torch.from_numpy(pred_lin).unsqueeze(0).unsqueeze(0), target_t)

        for key in METRIC_PAIRS:
            metrics[f"mod_{key}"].append(mod_scores[key])
            metrics[f"lin_{key}"].append(lin_scores[key])

        if mod_scores["ssim"] > best["ssim"]:
            best = {"ssim": mod_scores["ssim"], "pred": pred_mod, "linear": pred_lin,
                    "gt": n30, "n0": n0, "n60": n60, "triplet": triplet}
        if mod_scores["ssim"] < worst["ssim"]:
            worst = {"ssim": mod_scores["ssim"], "pred": pred_mod, "linear": pred_lin,
                     "gt": n30, "n0": n0, "n60": n60, "triplet": triplet}

    summary = {}
    for key in METRIC_KEYS:
        arr = np.asarray(metrics[key], dtype=np.float64)
        unit = 1000.0 if key.endswith("mse") else 1.0
        summary[key] = {"mean": float(arr.mean() * unit), "std": float(arr.std() * unit)}

    return {"metrics": metrics, "summary": summary, "best": best, "worst": worst,
            "n_val": len(val_triplets)}


def print_metrics_table(results):
    summary = results["summary"]
    header = (f"\n{'Metric':<14}{'Linear Interp':>22}{'Model Pred':>22}{'Delta':>10}")
    line = "-" * len(header)
    rows = []
    for metric, unit in (("ssim", ""), ("psnr", " dB"), ("mse", " (x10^-3)"), ("fsim", "")):
        lin = summary[f"lin_{metric}"]
        mod = summary[f"mod_{metric}"]
        delta = mod["mean"] - lin["mean"]
        rows.append(
            f"{metric + unit:<14}"
            f"{lin['mean']:>12.3f} +/- {lin['std']:<5.3f}"
            f"{mod['mean']:>12.3f} +/- {mod['std']:<5.3f}"
            f"{delta:>+10.3f}"
        )
    table = "\n".join([header, line] + rows)
    print(table)
    return table
