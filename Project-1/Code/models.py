import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from Code.metrics import mse, r2

def predict_centered(X, theta, y_mean):
    """Add back train-mean to restore intercept."""
    return X @ np.asarray(theta).ravel() + float(y_mean)

# ---------------- OLS via scikit-learn (no intercept; y is centered) ----------------
def fit_ols(X, y_c):
    """
    OLS with scikit-learn. We already centered y on train and scaled X,
    so we fit with no intercept and return theta (coef vector) to keep
    the same interface as before.
    """
    model = LinearRegression(fit_intercept=False, copy_X=True)
    model.fit(X, y_c)
    return model.coef_.ravel()

# ---------------- Ridge via scikit-learn (no intercept; y is centered) --------------
def fit_ridge(X, y_c, lam, n_factor=True, seed=42):
    """
    Ridge with scikit-learn. Keep your α = λ/n option when n_factor=True.
    Return theta to preserve your old predict_centered() usage.
    """
    n, p = X.shape
    alpha = (lam / n) if n_factor else lam
    model = Ridge(alpha=alpha, fit_intercept=False, seed=seed)
    model.fit(X, y_c)
    return model.coef_.ravel()

# ---------------- Lasso via scikit-learn (no intercept; y is centered) --------------
def fit_lasso(X, y_c, lam, seed=42, max_iter=20000):
    """
    Lasso with scikit-learn's coordinate descent (what your teacher wants).
    """
    model = Lasso(alpha=lam, fit_intercept=False, seed=seed, max_iter=max_iter)
    model.fit(X, y_c)
    return model.coef_.ravel()

# ---------------- Sweeps that your notebooks call (unchanged signatures) ------------
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
        theta = fit_ridge(Xtr, y_tr_c, lam, n_factor=n_factor)
        yhat = predict_centered(Xte, theta, y_mean)
        mses.append(mse(y_te, yhat))
        r2s.append(r2(y_te, yhat))
        norms.append(float(np.linalg.norm(theta)))
    return np.asarray(mses), np.asarray(r2s), np.asarray(norms)




#OLD

""" "OLS with closed-form solution (normal equations)."
def fit_ols(X, y_c):
    # Closed-form OLS (no intercept; y is centered)
    XT_X = X.T @ X # X^T X
    XT_y = X.T @ y_c # X^T y_c
    return np.linalg.solve(XT_X, XT_y) # θ = (X^T X)^{-1} X^T y_c

"Ridge regression with closed-form solution."
def fit_ridge(X, y_c, lam, n_factor=True):
    # Ridge: (X^T X + αI)^{-1} X^T y_c
    n, p = X.shape  # n samples, p features
    alpha = lam * (1.0/n if n_factor else 1.0) # α = λ/n or α = λ
    XT_X = X.T @ X # X^T X
    XT_y = X.T @ y_c # X^T y_c
    return np.linalg.solve(XT_X + alpha * np.eye(p), XT_y) # θ_ridge

def fit_ols_sklearn(X, y):
    ""OLS via scikit-learn (normal equations / least squares).""
    model = LinearRegression(fit_intercept=True, copy_X=True)
    model.fit(X, y)
    return model  # use model.predict(Xnew)

def fit_ridge_sklearn(X, y, lam, n_factor=False, n=None, random_state=42):
    ""Ridge via scikit-learn. Note: sklearn uses alpha directly (no /n).""
    alpha = (lam / n) if (n_factor and n is not None) else lam
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
    model.fit(X, y)
    return model  # use model.predict(Xnew)

"Prediction functions."
#Difference is shape between the two below
def predict_centered(X, theta, y_mean):   
    return X @ theta + y_mean # add back y_mean

#1D 
def predict_from_theta(X, theta, y_mean):
    return (X @ theta).ravel() + y_mean # add back y_mean

"Sweep functions to evaluate models over hyperparameter ranges."
def sweep_degree(X_full, y, split_func, deg_max=15):
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y) # split and scale
    degrees = range(1, deg_max + 1) # degrees to try
    mses, r2s, norms = [], [], [] # store metrics
    for p in degrees: 
        Xtr, Xte = X_tr_s[:, :p], X_te_s[:, :p] # use first p features  
        theta = fit_ols(Xtr, y_tr_c) # fit OLS
        yhat = predict_centered(Xte, theta, y_mean) # predict on test set
        mses.append(mse(y_te, yhat)) # compute metrics
        r2s.append(r2(y_te, yhat)) # compute metrics
        norms.append(float(np.linalg.norm(theta))) # L2 norm of theta
    return list(degrees), np.asarray(mses), np.asarray(r2s), np.asarray(norms) 

"Ridge sweep over lambda values."
def sweep_ridge(X_full, y, split_func, degree, lambdas, n_factor=True):
    X_tr_s, X_te_s, y_tr_c, y_te, _, y_mean = split_func(X_full, y) # split and scale
    Xtr, Xte = X_tr_s[:, :degree], X_te_s[:, :degree] # use first 'degree' features
    mses, r2s, norms = [], [], [] # store metrics
    for lam in np.asarray(lambdas): 
        theta = fit_ridge(Xtr, y_tr_c, lam, n_factor=n_factor) # fit Ridge
        yhat = predict_centered(Xte, theta, y_mean) # predict on test set
        mses.append(mse(y_te, yhat)) # compute metrics
        r2s.append(r2(y_te, yhat)) # compute metrics
        norms.append(float(np.linalg.norm(theta))) # L2 norm of theta
    return np.asarray(mses), np.asarray(r2s), np.asarray(norms)  """

