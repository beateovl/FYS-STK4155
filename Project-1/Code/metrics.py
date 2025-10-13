"""
metrics.py — common metrics and diagnostics.
"""
import numpy as np

#Mean squared error
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2)) # Mean Squared Error

#R² score
def r2(y_true, y_pred):
    mu = np.mean(y_true) # mean of y_true
    return float(1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mu)**2)) # R²

#L2 norm of parameter vector
def l2_norm(theta):
    return float(np.linalg.norm(theta)) # L2 norm of parameter vector

#Condition number of feature matrix that indicates multicollinearity
def condition_number(X):
    s = np.linalg.svd(X, compute_uv=False) # singular values
    return float(s.max() / s.min())     # cond(X) = σ_max / σ_min
