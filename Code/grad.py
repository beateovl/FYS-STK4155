# grad.py — gradient descent for OLS / Ridge

import numpy as np

def grad_ols(X, y, theta):
    """OLS gradient, 1/2n."""
    y = np.asarray(y, float).ravel()
    theta = np.asarray(theta, float).ravel()
    r = X @ theta - y
    return (X.T @ r) / X.shape[0]

def grad_ridge(X, y, theta, lam):
    """Ridge gradient, 1/2n."""
    return grad_ols(X, y, theta) + lam * theta

def gd(X, y, eta, iters, theta0=None, lam=None):
    """
    Plain gradient descent.
    - lam=None  → OLS
    - lam=float → Ridge with that λ
    Returns θ:(p,)
    """
    p = X.shape[1]
    theta = np.zeros(p, dtype=float) if theta0 is None else np.asarray(theta0, float).ravel()
    for _ in range(iters):
        g = grad_ols(X, y, theta) if lam is None else grad_ridge(X, y, theta, lam)
        theta -= eta * g
    return theta
