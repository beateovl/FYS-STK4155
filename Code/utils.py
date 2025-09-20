"""
utils.py — misc helpers for reproducibility and saving artifacts.
"""

from __future__ import annotations
import json, os
import numpy as np
from datetime import datetime


def set_seed(seed: int = 42):
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamped(fname: str, ext: str = ".png") -> str:
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{fname}_{t}{ext}"


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
