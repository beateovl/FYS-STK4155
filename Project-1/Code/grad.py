# grad.py — gradient descent for OLS / Ridge

import numpy as np


def grad_ols(X, y, theta):
    """OLS gradient, 1/2n.  """
    y = np.asarray(y, float).ravel() # ensure y is a 1D array
    theta = np.asarray(theta, float).ravel() # ensure theta is a 1D array
    r = X @ theta - y # residuals
    return (X.T @ r) / X.shape[0] # 1/2n factor cancels with 2 from derivative

def grad_ridge(X, y, theta, lam): 
    """Ridge gradient, 1/2n. lam = λ."""
    return grad_ols(X, y, theta) + lam * theta  # 1/2n factor cancels with 2 from derivative

def gd(X, y, eta, iters, theta0=None, lam=None):
    """
    Plain gradient descent.
    - lam=None  → OLS
    - lam=float → Ridge with that λ
    Returns θ:(p,)
    eta: learning rate
    """
    p = X.shape[1] # n features
    theta = np.zeros(p, dtype=float) if theta0 is None else np.asarray(theta0, float).ravel() # ensure 1D
    for _ in range(iters): 
        g = grad_ols(X, y, theta) if lam is None else grad_ridge(X, y, theta, lam) # gradient 
        theta -= eta * g # gradient step
    return theta 
