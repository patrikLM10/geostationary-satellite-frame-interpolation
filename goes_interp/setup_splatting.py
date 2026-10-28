"""Softmax-splatting (sniklaus) CUDA kernel setup with graceful fallback.

The high-performance CUDA forward-splatting kernel comes from
https://github.com/sniklaus/softmax-splatting (cloned on first run).
If the CUDA build is unavailable, the pipeline transparently falls back
to a pure-PyTorch backward-warp approximation.
"""

import os
import subprocess
import sys

import torch

from goes_interp import config

SPLAT_CUDA = False
_SOFTSPLATTING_URL = "https://github.com/sniklaus/softmax-splatting.git"


def ensure() -> bool:
    """Clone (once) and test the softmax-splatting CUDA kernel. Returns True when available."""
    global SPLAT_CUDA
    cfg_dir = config.SOFTMAX_SPLATTING_DIR
    if not cfg_dir.exists():
        print("Cloning softmax-splatting repository...")
        subprocess.run(
            ["git", "clone", "--quiet", _SOFTSPLATTING_URL, str(cfg_dir)],
            check=False,
        )
    splatting_path = str(cfg_dir)
    if splatting_path not in sys.path:
        sys.path.append(splatting_path)

    SPLAT_CUDA = False
    if torch.cuda.is_available():
        try:
            import softsplat

            dummy_input = torch.ones(1, 1, 4, 4, device="cuda")
            dummy_flow = torch.ones(1, 2, 4, 4, device="cuda")
            softsplat_splat(dummy_input, dummy_flow, dummy_input)
            SPLAT_CUDA = True
            print("Softmax splatting CUDA build successful.")
        except Exception as exc:  # noqa: BLE001
            print(f"Softmax splatting CUDA build failed: {exc}\nUsing pure-PyTorch fallback.")
    else:
        print("CUDA not available; using pure-PyTorch fallback splatting.")
    return SPLAT_CUDA


def softsplat_splat(ten_input, ten_flow, ten_metric):
    """Call the kernel across both known versions of the softsplat API."""
    import softsplat

    if hasattr(softsplat, "softsplat"):
        return softsplat.softsplat(ten_input, ten_flow, ten_metric, "softmax")
    return softsplat.FunctionSoftsplat(
        tenInput=ten_input, tenFlow=ten_flow, tenMetric=ten_metric, strType="softmax"
    )
