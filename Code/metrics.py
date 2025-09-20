"""
metrics.py — common metrics and diagnostics.
"""
import numpy as np


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    mu = y_true.mean()
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mu) ** 2))


def l2_norm(theta):
    return float(np.linalg.norm(theta))


def condition_number(X):
    """2-norm condition number of X (or of XᵀX if you pass that in)."""
    s = np.linalg.svd(X, compute_uv=False)
    return float(s.max() / s.min())
