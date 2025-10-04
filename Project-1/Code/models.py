import numpy as np
from Code.metrics import mse, r2  

"OLS with closed-form solution (normal equations)."
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
    return np.asarray(mses), np.asarray(r2s), np.asarray(norms) 

