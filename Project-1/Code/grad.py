# Code for gradients and gradient descent (GD).
import numpy as np

# ---------- GD ----------
def grad_ols(X, y, theta):
    """OLS gradient for loss (1/2n)||Xθ - y||^2."""
    y = np.asarray(y, float).ravel()
    theta = np.asarray(theta, float).ravel()
    r = X @ theta - y
    return (X.T @ r) / X.shape[0]


def grad_ridge(X, y_c, theta, lam, n_factor=True):
    n = X.shape[0]
    alpha = (lam / n) if n_factor else lam
    r = X @ theta - y_c
    return (X.T @ r) / n + alpha * theta

# --------------- Lasso GD ------------------
# Lasso via gradient descent (fixed step)
def lasso_gradient(X, y_c, lam, iters=5000, eta=1e-3, theta0=None):
    n, p = X.shape 
    theta = np.zeros(p) if theta0 is None else theta0.copy() #.copy() to avoid modifying input
    for _ in range(iters): 
        r = X @ theta - y_c
        grad = (2.0/n) * (X.T @ r) + lam * np.sign(theta)
        theta -= eta * grad
    return theta

# ---------- loss helpers, to show convergence ----------
# Convert lambda to alpha for Ridge
def alpha_from_lambda(lam, n, n_factor):
    return (lam / n) if (n_factor and lam is not None) else (lam or 0.0)

# OLS loss
def loss_ols(X, y, theta):
    r = X @ theta - y
    return 0.5 * (r @ r) / X.shape[0]

# Ridge loss
def loss_ridge(X, y, theta, lam, n_factor=True):
    n = X.shape[0]
    alpha = (lam / n) if n_factor else lam
    r = X @ theta - y
    return 0.5 * (r @ r) / n + 0.5 * alpha * (theta @ theta) 



