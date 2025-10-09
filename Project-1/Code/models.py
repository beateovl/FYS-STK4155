import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from Code.metrics import mse, r2

def predict_centered(X, theta, y_mean):
    """Add back train-mean to restore intercept."""
    return X @ np.asarray(theta).ravel() + float(y_mean)

# ---------------- OLS via scikit-learn (no intercept; y is centered) ----------------
def fit_ols(X, y_c):
    """
    OLS with scikit-learn. 
    """
    model = LinearRegression(fit_intercept=False, copy_X=True)
    model.fit(X, y_c)
    return model.coef_.ravel()

# ---------------- Ridge own code and via scikit-learn (no intercept; y is centered) --------------

def fit_ridge(X, y_c, lam, n_factor=True):
    """
    Ridge solution for
        J(θ) = (1/(2n))‖Xθ - y_c‖^2 + (α/2)‖θ‖^2

    If n_factor=True (default): `lam` is λ and we set α = λ/n internally.
        Solve: (X^T X + λ I) θ = X^T y_c
    If n_factor=False: `lam` is α directly.
        Solve: (X^T X / n + α I) θ = (X^T y_c) / n
    """
    X = np.asarray(X, float)
    y = np.asarray(y_c, float).ravel()
    n, p = X.shape
    I = np.eye(p, dtype=float)

    if n_factor:             # lam is λ
        A = X.T @ X + lam * I
        b = X.T @ y
    else:                    # lam is α
        A = (X.T @ X) / n + lam * I
        b = (X.T @ y) / n

    return np.linalg.solve(A, b)


def fit_ridge_sklearn(X, y, lam, n_factor=False, n=None, random_state=42):
    """Ridge via scikit-learn. Note: sklearn uses alpha directly (no /n)."""
    alpha = (lam / n) if (n_factor and n is not None) else lam
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
    model.fit(X, y)
    return model  # use model.predict(Xnew)

# ---------------- Lasso via scikit-learn (no intercept; y is centered) --------------

def fit_lasso_skl(X, y_c, lam, max_iter=20000, tol=1e-6, selection="cyclic"):
    # Our loss: (1/n)||r||^2 + lam * ||theta||_1
    # sklearn loss: (1/(2n))||r||^2 + alpha * ||theta||_1
    alpha = lam / 2.0
    model = Lasso(alpha=alpha,
                  fit_intercept=False,
                  max_iter=max_iter,
                  tol=tol,
                  selection=selection)
    model.fit(X, y_c)
    return model.coef_.ravel()

# ---------------- Sweeps (unchanged signatures) ------------
def sweep_degree(X_full, y, split_func, deg_max):
    """
    OLS vs polynomial degree (hold-out split inside split_func).
    Returns: degrees, test_mse, test_r2, l2_norm(theta)
    """
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)
    degrees = range(1, deg_max + 1)
    mses, r2s, norms = [], [], []
    for d in degrees:
        Xtr, Xte = X_tr_s[:, :d], X_te_s[:, :d]
        theta = fit_ols(Xtr, y_tr_c)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(float(np.linalg.norm(theta)))
    return list(degrees), np.asarray(mses), np.asarray(r2s), np.asarray(norms)

def sweep_ridge(X_full, y, split_func, degree, lambdas, n_factor=True):
    """
    Ridge vs lambda for a fixed degree (hold-out split inside split_func).
    Returns: test_mse, test_r2, l2_norm(theta)
    """
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y)
    Xtr, Xte = X_tr_s[:, :degree], X_te_s[:, :degree]
    mses, r2s, norms = [], [], []
    for lam in np.asarray(lambdas):
        theta = fit_ridge(Xtr, y_tr_c, lam)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(float(np.linalg.norm(theta)))
    return np.asarray(mses), np.asarray(r2s), np.asarray(norms)


