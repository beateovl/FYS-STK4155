"""
plots.py — plotting helpers (no math here).
"""

import matplotlib.pyplot as plt

def plot_mse_r2_vs_degree(degrees, mse_vals, r2_vals, title="OLS vs Degree", savepath=None):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    l1, = ax1.plot(degrees, mse_vals, 'o-', color='C0', label='Test MSE')
    ax1.set_xlabel('Polynomial degree')
    ax1.set_ylabel('MSE (test)', color='C0'); ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, ls='--', alpha=0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(degrees, r2_vals, 's--', color='C2', label='Test R²')
    ax2.set_ylabel('R² (test)', color='C2'); ax2.tick_params(axis='y', colors='C2')

    ax1.legend([l1, l2], ['Test MSE', 'Test R²'], loc='best')

    # Key changes:
    ax1.set_title(title, pad=8)          # set on ax1, not plt.title
    fig.tight_layout()                   # or: fig.tight_layout(rect=[0,0,1,0.96])

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)

    plt.show()


def plot_theta_norms(x, norms, xlabel, title=r'Coefficient size $\|\theta\|_2$', savepath=None):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, norms, 'r-x')
    plt.xlabel(xlabel); plt.ylabel(r'$\|\theta\|_2$')
    plt.title(title); plt.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.show()


def plot_ridge_curves(lambdas, mse_vals, r2_vals, title="Ridge vs λ", savepath=None):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    l1, = ax1.plot(lambdas, mse_vals, 'o-', color='C0', label='Test MSE (Ridge)')
    ax1.set_xscale('log')
    ax1.set_xlabel('λ (log scale)')
    ax1.set_ylabel('MSE (test)', color='C0'); ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, ls='--', alpha=0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(lambdas, r2_vals, 's--', color='C2', label='Test R² (Ridge)')
    ax2.set_ylabel('R² (test)', color='C2'); ax2.tick_params(axis='y', colors='C2')

    ax1.legend([l1, l2], ['Test MSE', 'Test R²'], loc='best')

    ax1.set_title(title, pad=8)          # set on ax1
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.show()
