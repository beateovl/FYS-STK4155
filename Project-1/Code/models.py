import numpy as np
from Code.metrics import mse, r2  

def fit_ols(X, y_c):
    # Closed-form OLS (no intercept; y is centered)
    XT_X = X.T @ X
    XT_y = X.T @ y_c
    return np.linalg.solve(XT_X, XT_y)

def fit_ridge(X, y_c, lam, n_factor=True):
    # Ridge: (X^T X + αI)^{-1} X^T y_c
    n, p = X.shape
    alpha = lam * (1.0/n if n_factor else 1.0)
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y_c)

def predict_centered(X, theta, y_mean):
    return X @ theta + y_mean

def sweep_degree(X_full, y, split_func, deg_max=15):
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)
    degrees = range(1, deg_max + 1)
    mses, r2s, norms = [], [], []
    for p in degrees:
        Xtr, Xte = X_tr_s[:, :p], X_te_s[:, :p]
        theta = fit_ols(Xtr, y_tr_c)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(float(np.linalg.norm(theta)))
    return list(degrees), np.asarray(mses), np.asarray(r2s), np.asarray(norms)

def sweep_ridge(X_full, y, split_func, degree, lambdas, n_factor=True):
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)
    Xtr, Xte = X_tr_s[:, :degree], X_te_s[:, :degree]
    mses, r2s, norms = [], [], []
    for lam in np.asarray(lambdas):
        theta = fit_ridge(Xtr, y_tr_c, lam, n_factor=n_factor)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(float(np.linalg.norm(theta)))
    return np.asarray(mses), np.asarray(r2s), np.asarray(norms)
