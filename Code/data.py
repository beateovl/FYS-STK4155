"""
data.py — data generation, feature building, and leakage-safe split/scale.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def runge(x):
    """
    Runge function, vectorized: f(x) = 1/(1 + 25 x^2).
    Parameters
    ----------
    x : array, shape (n,1) or (n,)
    Returns
    -------
    y : array, shape like x
    """
    x = np.asarray(x, float)
    return 1.0 / (1.0 + 25.0 * x**2)


def make_data(n=100, noise_sd=0.1, seed=42):
    """
    Create (x,y) on [-1,1] with Gaussian noise added to Runge.
    Returns x:(n,1), y:(n,1)
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = runge(X) + rng.normal(0.0, noise_sd, size=X.shape)
    return X, y


def build_features(X, degree, include_bias=False):
    """
    Polynomial design matrix [x, x^2, ..., x^degree].
    """
    return PolynomialFeatures(degree=degree, include_bias=include_bias).fit_transform(x)


def split_and_scale(X, y, test_size=0.2, random_state=42, center_y=True):
    """
    Splits data into train and test (20%). Centering y is optional if not wanting to penalize
    the intercept. Column-wise mean and std from X_tr. 
    Leakage safe. 
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_tr)                  # fit on TRAIN only (no leakage)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    if center_y:
        y_mean = y_tr.mean()
        y_tr_c = y_tr - y_mean
        return X_tr_s, X_te_s, y_tr_c, y_te, scaler, y_mean
    return X_tr_s, X_te_s, y_tr, y_te, scaler, None