"""Reproducibility and small tensor helpers."""

import random

import numpy as np
import torch

from goes_interp import config


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(array, device=None):
    """(H, W) numpy array -> (1, 1, H, W) float tensor, optionally on device."""
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).unsqueeze(0).float()
    return tensor.to(device) if device is not None else tensor


def describe_device():
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
