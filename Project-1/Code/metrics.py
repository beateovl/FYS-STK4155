"""
metrics.py — common metrics and diagnostics.
"""
import numpy as np

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2)) # Mean Squared Error

def r2(y_true, y_pred):
    mu = np.mean(y_true) # mean of y_true
    return float(1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mu)**2)) # R²


def l2_norm(theta):
    return float(np.linalg.norm(theta)) # L2 norm of parameter vector


def condition_number(X):
    """2-norm condition number of X (or of XᵀX if you pass that in)."""
    s = np.linalg.svd(X, compute_uv=False) # singular values
    return float(s.max() / s.min())     # cond(X) = σ_max / σ_min
