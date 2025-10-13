"""
plots.py — plotting helpers. Only some plots for part a and b, not used in parts c-f.
"""

import matplotlib.pyplot as plt

def plot_theta_norms(x, norms, xlabel,
                     title=r'Coefficient size $\|\theta\|_2$',
                     savepath=None, logx=False):
    fig = plt.figure(figsize=(8, 4))
    if logx: 
        plt.semilogx(x, norms, 'r-x')   # log x-axis
    else:
        plt.plot(x, norms, 'r-x') # linear x-axis
    plt.xlabel(xlabel)
    plt.ylabel(r'$\|\theta\|_2$')
    plt.title(title)
    plt.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.show()


"Ridge plot function. Shows MSE and R2 vs lambda."
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


"General OLS plot function. Shows MSE and R2 vs polynomial degree."
def plot_mse_r2_vs_degree(degrees, mse_vals, r2_vals, title="OLS vs Degree",
                          savepath=None, ax=None, show=True):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(9,5))
    else:
        ax1 = ax
        fig = ax1.figure

    l1, = ax1.plot(degrees, mse_vals, 'o-', color='C0', label='Test MSE')
    ax1.set_xlabel('Polynomial degree')
    ax1.set_ylabel('MSE (test)', color='C0'); ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, ls='--', alpha=0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(degrees, r2_vals, 's--', color='C1', label='Test $R^2$')
    ax2.set_ylabel(r'$R^2$ (test)', color='C1'); ax2.tick_params(axis='y', colors='C1')

    ax1.legend([l1, l2], ['Test MSE', r'Test $R^2$'], loc='best')
    ax1.set_title(title, pad=8)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show and (ax is None) and (savepath is None):
        plt.show()

