# Code for gradients and gradient descent (GD).
import numpy as np

# ---------- GD ----------
def grad_ols(X, y, theta):
    """OLS gradient for loss (1/2n)||Xθ - y||^2."""
    y = np.asarray(y, float).ravel()
    theta = np.asarray(theta, float).ravel()
    r = X @ theta - y
    return (X.T @ r) / X.shape[0]

def grad_ridge(X, y, theta, lam, n_factor=True):
    """
    Ridge gradient for loss (1/2n)||Xθ - y||^2 + (α/2)||θ||^2.
    If n_factor=True, interpret lam as λ and use α = λ/n (matches closed-form).
    """
    alpha = (lam / X.shape[0]) if n_factor else lam
    return grad_ols(X, y, theta) + alpha * theta

# --------------- Lasso GD ------------------
def lasso_gd(X, y, theta, eta, lmbda, Niterations):
 
    #LASSO via plain gradient descent with fixed learning rate.

    n = X.shape[0]  # number of samples
    for iter in range(Niterations):
        # 1) OLS gradient component: (2/n) * X^T (X theta - y)
        residual = X @ theta - y
        OLS_gradient_component = (2.0 / n) * (X.T @ residual)

        # 2) L1 gradient component (subgradient): lambda * sign(theta)
        L1_gradient_component = lmbda * np.sign(theta)

        # 3) Total gradient
        total_gradient = OLS_gradient_component + L1_gradient_component

        # 4) Parameter update with fixed learning rate
        theta -= eta * total_gradient

    return theta 

def soft_threshold(z, tau):
    # elementwise shrinkage: sign(z)*max(|z|-tau, 0)
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

def lasso_ista(X, y, lam, eta, iters=5000, theta0=None, early_stop=None):
    """
    Minimizes: (1/n)||Xθ - y||^2 + lam * ||θ||_1
    ISTA step: θ ← S_{ηλ}(θ - η ∇f),  ∇f = (2/n) X^T(Xθ - y)
    """
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    losses = []
    for t in range(iters):
        r = X @ theta - y
        grad = (2.0/n) * (X.T @ r)
        theta = soft_threshold(theta - eta*grad, eta*lam)
        if early_stop is not None:
            # compute *validation* loss if provided
            L_val = early_stop(theta)
            losses.append(L_val)
            # plateau rule: compare window means
            W = 300
            if len(losses) >= 2*W:
                new = np.mean(losses[-W:])
                old = np.mean(losses[-2*W:-W])
                if (old - new) <= 1e-4*max(1.0, old):
                    break
    return theta

# ---------- loss helpers (consistent with the grads above) ----------
def alpha_from_lambda(lam, n, n_factor):
    return (lam / n) if (n_factor and lam is not None) else (lam or 0.0)

def loss_ols(X, y, theta):
    r = X @ theta - y
    return 0.5 * (r @ r) / X.shape[0]

""" def loss_ridge(X, y, theta, lam, n_factor=True):
    n = X.shape[0]
    alpha = alpha_from_lambda(lam, n, n_factor)
    r = X @ theta - y
    return 0.5 * (r @ r) / n + 0.5 * alpha * (theta @ theta) """

def loss_ridge(X, y, theta, lam, n_factor=True):
    n = X.shape[0]
    alpha = (lam / n) if n_factor else lam
    r = X @ theta - y
    return 0.5 * (r @ r) / n + 0.5 * alpha * (theta @ theta)

# ---------- GD supporting loss tracking ----------
def gd(X, y, eta, iters, theta0=None, lam=None, n_factor=True,
       track_loss=False, every=1):
    """
    Plain gradient descent (fixed step).
      - lam=None  → OLS on loss (1/2n)||Xθ - y||^2
      - lam=float → Ridge on loss (1/2n)||Xθ - y||^2 + (α/2)||θ||^2
                    with α=λ/n if n_factor=True (matches your closed form)

    If track_loss=True:
      returns (theta, losses) where 'losses' records the training loss
      every 'every' iterations and once at the end.
    Otherwise:
      returns theta.
    """
    p = X.shape[1]
    theta = np.zeros(p, dtype=float) if theta0 is None else np.asarray(theta0, float).ravel()
    losses = [] if track_loss else None

    for t in range(iters):
        if lam is None:
            g = grad_ols(X, y, theta)
            if track_loss and (t % every) == 0:
                losses.append(loss_ols(X, y, theta))
        else:
            g = grad_ridge(X, y, theta, lam, n_factor=n_factor)
            if track_loss and (t % every) == 0:
                losses.append(loss_ridge(X, y, theta, lam, n_factor=n_factor))

        theta -= eta * g

    if track_loss:
        # ensure final loss is recorded
        if lam is None:
            losses.append(loss_ols(X, y, theta))
        else:
            losses.append(loss_ridge(X, y, theta, lam, n_factor=n_factor))
        return theta, np.asarray(losses)

    return theta
# ---------- Mini-batch SGD with several optimizers + loss tracking ----------
