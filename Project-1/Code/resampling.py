"""
resampling.py — bootstrap + k-fold utilities.
"""
import numpy as np
from sklearn.model_selection import KFold


def bootstrap_predictions(fit_fn, pred_fn, Xtr_s, ytr_c, Xte_s, y_mean, B=300, seed=42):
    """Return P (B, n_test) of bootstrap predictions on a fixed test set."""
    rng = np.random.default_rng(seed)
    n = Xtr_s.shape[0]
    idxs = rng.integers(0, n, size=(B, n)) 
    # Each row of idxs is a bootstrap sample of indices into (Xtr_s, ytr_c).
    return np.vstack([
        pred_fn(Xte_s, fit_fn(Xtr_s[idx], ytr_c[idx]), y_mean).ravel()
        for idx in idxs
    ])


def bias_variance_from_preds(P, y_true):
    """Given P:(B,n) and noise-free y_true:(n,), return (bias^2, var, empirical MSE)."""
    m = P.mean(axis=0)
    bias2 = float(((m - y_true) ** 2).mean())
    var = float(P.var(axis=0).mean())
    mse = float(((P - y_true) ** 2).mean())
    return bias2, var, mse


def kfold_cv(X_full, y, degree, k, fit_fn, pred_fn, splitter=None):
    """k-fold CV MSE for a fixed degree. Scaling/centering done per fold. 
    """
    # Create a KFold splitter if none given
    splitter = splitter or KFold(n_splits=k, shuffle=True, random_state=42) 
    mses = []
    # Loop over each train/test split
    for tr, te in splitter.split(X_full): 
        Xtr, Xte = X_full[tr], X_full[te]
        ytr, yte = y[tr], y[te]

        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        sd = np.where(sd == 0, 1, sd)
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

        y_mean = ytr.mean()
        ytr_c = ytr - y_mean

        Xtr_p, Xte_p = Xtr_s[:, :degree], Xte_s[:, :degree]
        theta = fit_fn(Xtr_p, ytr_c)
        yhat = pred_fn(Xte_p, theta, y_mean)

        mses.append(float(((yte - yhat) ** 2).mean()))
    return float(np.mean(mses)) # average MSE over folds

