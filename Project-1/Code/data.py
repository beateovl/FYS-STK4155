import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

"Synthetic dataset functions for the Runge regression experiments."

def make_data(n=300, noise_sd=0.3, seed=42): # Runge function + noise
    rng = np.random.default_rng(seed) 
    x = rng.uniform(-1, 1, size=n)  # Generate random x values
    # Runge func: 
    f = 1 / (1 + 25 * x**2)
    y = f + rng.normal(0, noise_sd, size=n) # Add Gaussian noise
    return x, y

def build_features(X, degree, include_bias=False): # polynomial features
    X = np.asarray(X).reshape(-1, 1)  # ensure 2D
    return PolynomialFeatures(degree=degree, include_bias=include_bias).fit_transform(X)

def split_and_scale(X, y, test_size=0.2, random_state=42, center_y=True): # split, scale, center
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state) # split
    scaler = StandardScaler().fit(X_tr)     # fit on TRAIN only
    X_tr_s = scaler.transform(X_tr) # scale TRAIN
    X_te_s = scaler.transform(X_te) # scale TEST
    y_mean = float(np.mean(y_tr)) if center_y else 0.0 # center TRAIN only
    y_tr_c = y_tr - y_mean # center TRAIN
    return X_tr_s, X_te_s, y_tr_c, y_te, scaler, y_mean # return scaler + y_mean for inverse transform if needed
