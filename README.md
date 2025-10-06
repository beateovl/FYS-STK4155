# FYS-STK4155

Group members: Selma Beate Øvland 

Project description: Studying regression methods on the Runge function


<pre>
Project-1/
├── Code/
│   ├── data.py          # make_data, build_features, split_and_scale
│   ├── models.py        # OLS/Ridge/Lasso (sklearn), sweeps, predict helpers
│   ├── metrics.py       # mse, r2, (optionally) condition_number
│   ├── plots.py         # plotting helpers (ridge curves, θ-norms, etc.)
│   ├── grad.py          # gradient functions + full-batch optimizers
│   ├── resampling.py    # hold-out/CV/bootstrap helpers
├── Notebooks/
│   ├── Part_a_OLS_degree.ipynb          # Part (a): OLS vs degree
│   ├── Part_b_Ridge_lambda.ipynb        # Part (b): Ridge vs λ
│   ├── Part_c_d_e_f_GD.ipynb            # Parts (c–f): GD/Momentum/AdaGrad/RMSProp/Adam + SGD
│   ├── Part_g_BiasVariance.ipynb        # Part (g) Bias–Variance trade-off
│   └── Part_h_CrossValidation.ipynb     # Part (h) K-fold Cross-Validation
├── Figures/                              # exported figures (auto-created)
├── README.md                             # this file
└── Project1.pdf                          # assignment text (reference)
</pre>




Environment & setup:
Python 3.10+ (tested with 3.12)

