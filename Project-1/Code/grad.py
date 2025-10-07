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

"Loss functions for tracing."

def loss_ols(X, y_c, theta):
    r = X @ theta - y_c 
    return 0.5 * (r @ r) / X.shape[0]

def loss_ridge(X, y_c, theta, lam, n_factor=True):
    n = X.shape[0]
    alpha = (lam / n) if n_factor else lam
    r = X @ theta - y_c
    return 0.5 * (r @ r) / n + 0.5 * alpha * (theta @ theta) 

def alpha_from_lambda(lam, n, n_factor=True):
    # Ridge convention: α = λ/n if n_factor=True, else α = λ
    return (lam / n) if n_factor else lam
