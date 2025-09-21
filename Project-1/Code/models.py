# models.py — closed-form OLS/Ridge + tiny sweep helpers

import numpy as np

# ---------- Core solvers ----------
def fit_ols(X, y):
    """OLS estimator.  (y centered; no explicit intercept)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]

def fit_ridge(X, y, lam, n_factor=True):
    """
    Ridge estimator. 
    """
    n, p = X.shape
    alpha = n * lam if n_factor else lam
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)

def predict_centered(X, theta, y_mean):
    """Restore intercept after training on centered y."""
    return X @ theta + y_mean

# ---------- Metrics ----------
def mse(y, yhat): return float(np.mean((y - yhat) ** 2))

def r2(y, yhat):
    mu = y.mean()
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - mu) ** 2))

# ---------- Experiment sweeps ----------
def sweep_degree(X_full, y, split_func, deg_max=15):
    """
    Estimator: OLS. Calls split/scale func to get data and y_mean. 
    Poly-degree varies. Looks at how # of degrees affects fit.
    """
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)

    degrees = range(1, deg_max + 1)
    mses, r2s, norms = [], [], []

    for p in degrees:
        Xtr, Xte = X_tr_s[:, :p], X_te_s[:, :p]
        theta = fit_ols(Xtr, y_tr_c)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(np.linalg.norm(theta))

    return list(degrees), np.asarray(mses), np.asarray(r2s), np.asarray(norms)

def sweep_ridge(X_full, y, split_func, degree, lambdas, n_factor=True):
    """
    Estimator: Ridge. Fixed poly-degree. Looks at how lambda affects fit. 
    Returns arrays: MSEs, R2s, ||θ||.
    """
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)
    Xtr, Xte = X_tr_s[:, :degree], X_te_s[:, :degree]

    mses, r2s, norms = [], [], []

    for lam in np.asarray(lambdas):
        theta = fit_ridge(Xtr, y_tr_c, lam, n_factor=n_factor)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(np.linalg.norm(theta))

    return np.asarray(mses), np.asarray(r2s), np.asarray(norms)
