# grad.py — gradient descent for OLS / Ridge

import numpy as np


def grad_ols(X, y, theta):
    """OLS gradient, 1/2n.  """
    y = np.asarray(y, float).ravel() # ensure y is a 1D array
    theta = np.asarray(theta, float).ravel() # ensure theta is a 1D array
    r = X @ theta - y # residuals
    return (X.T @ r) / X.shape[0] # 1/2n factor cancels with 2 from derivative

def grad_ridge(X, y, theta, lam, n_factor=True): 
    """Ridge gradient, 1/2n. lam = λ.
    If n_factor=True, interpret lam as λ and use α = λ/n to match closed-form.
    """
    alpha = (lam / X.shape[0]) if n_factor else lam
    return grad_ols(X, y, theta) + alpha * theta  # 1/2n factor cancels with 2 from derivative

def gd(X, y, eta, iters, theta0=None, lam=None, n_factor=True):
    """
    Plain gradient descent.
    - lam=None  → OLS
    - lam=float → Ridge with that λ
    Returns θ:(p,)
    eta: learning rate
    """
    p = X.shape[1]
    theta = np.zeros(p, dtype=float) if theta0 is None else np.asarray(theta0, float).ravel()
    for _ in range(iters):
        if lam is None:
            g = grad_ols(X, y, theta)
        else:
            g = grad_ridge(X, y, theta, lam, n_factor=n_factor)
        theta -= eta * g
    return theta

def compute_alpha(lam, n, n_factor):
    """Helper to get α from λ, consistent with grad_ridge."""
    return (lam / n) if (n_factor and lam is not None) else (lam or 0.0)

def loss_ols(X, y, theta):
    """OLS loss, 1/2n."""
    r = X @ theta - y
    return 0.5 * (r @ r) / X.shape[0]

def loss_ridge(X, y, theta, lam, n_factor=True):
    """Ridge loss, 1/2n. lam = λ."""
    n = X.shape[0]
    alpha = _alpha_from_lambda(lam, n, n_factor)
    r = X @ theta - y
    return 0.5 * (r @ r) / n + 0.5 * alpha * (theta @ theta)
