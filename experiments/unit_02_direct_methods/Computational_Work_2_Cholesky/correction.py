import numpy as np
import json
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)
os.makedirs('figures', exist_ok=True)

matricula = 318375

# --- 1. ORIGINAL FORMULATION ---
def gerar_matriz_por_matricula(matricula, n=11):
    np.random.seed(matricula)
    h = 1 / (n - 1)
    M = np.zeros((n, n))
    k = np.random.uniform(0.05, 0.3)
    fonte = np.random.uniform(0.5, 2.0)
    tipo_bc = np.random.choice([
        "neumann-dirichlet",
        "dirichlet-dirichlet",
        "neumann-neumann"
    ])
    coef = 1 / h**2
    s = np.ones(n) * (h**2 / k) * fonte
    
    for i in range(1, n - 1):
        M[i, i - 1] = coef
        M[i, i] = -2 * coef
        M[i, i + 1] = coef
        
    if tipo_bc == "neumann-dirichlet":
        M[0, 0] = -coef
        M[0, 1] = coef
        M[-1, -1] = 1
        s[-1] = 0
    elif tipo_bc == "dirichlet-dirichlet":
        M[0, 0] = 1
        M[-1, -1] = 1
        s[0] = 0
        s[-1] = 0
    elif tipo_bc == "neumann-neumann":
        M[0, 0] = -coef
        M[0, 1] = coef
        M[-1, -2] = -coef
        M[-1, -1] = coef
        
    return M, s, k, fonte, tipo_bc

M_orig, s_orig, k, f, bc = gerar_matriz_por_matricula(matricula)
h = 0.1
T_orig = np.linalg.solve(M_orig, s_orig)
r_orig = np.linalg.norm(s_orig - M_orig @ T_orig, 2)
rel_r_orig = r_orig / np.linalg.norm(s_orig) if np.linalg.norm(s_orig) > 0 else r_orig

# --- 2. CORRECTED FORMULATION ---
M_corr = np.zeros((11, 11))
s_corr = np.zeros(11)

coef = 1 / h**2
for i in range(1, 10):
    M_corr[i, i-1] = -coef
    M_corr[i, i] = 2 * coef
    M_corr[i, i+1] = -coef
    s_corr[i] = f / k

# BC x=0: T_0 = 0
M_corr[0, 0] = 1
s_corr[0] = 0

# BC x=1: T_10 = 0
M_corr[-1, -1] = 1
s_corr[-1] = 0

T_corr = np.linalg.solve(M_corr, s_corr)
r_corr = np.linalg.norm(s_corr - M_corr @ T_corr, 2)
rel_r_corr = r_corr / np.linalg.norm(s_corr) if np.linalg.norm(s_corr) > 0 else r_corr

# --- 3. EXACT SOLUTION ---
x_grid = np.linspace(0, 1, 11)
T_exact = (f / (2*k)) * x_grid * (1 - x_grid)

# --- 4. ERROR ANALYSIS ---
err_orig = np.abs(T_orig - T_exact)
err_corr = np.abs(T_corr - T_exact)

L2_err_orig = np.linalg.norm(err_orig)
L2_err_corr = np.linalg.norm(err_corr)

rel_L2_orig = L2_err_orig / np.linalg.norm(T_exact)
rel_L2_corr = L2_err_corr / np.linalg.norm(T_exact)

max_err_orig = np.max(err_orig)
max_err_corr = np.max(err_corr)

# --- 5. CONVERGENCE CHECK ---
def solve_corrected(N):
    hh = 1 / (N - 1)
    xx = np.linspace(0, 1, N)
    MM = np.zeros((N, N))
    ss = np.zeros(N)
    cc = 1 / hh**2
    for i in range(1, N - 1):
        MM[i, i-1] = -cc
        MM[i, i] = 2 * cc
        MM[i, i+1] = -cc
        ss[i] = f / k
    MM[0, 0] = 1
    ss[0] = 0
    MM[-1, -1] = 1
    ss[-1] = 0
    TT = np.linalg.solve(MM, ss)
    TT_ex = (f / (2*k)) * xx * (1 - xx)
    err = np.abs(TT - TT_ex)
    return np.max(err), np.linalg.norm(err)*np.sqrt(hh) 

errors_max = []
errors_L2 = []
Ns = [11, 21, 41, 81]
hs = [1/(N-1) for N in Ns]
for N in Ns:
    emax, el2 = solve_corrected(N)
    errors_max.append(emax)
    errors_L2.append(el2)

if max(errors_max) < 1e-12:
    rate_max = 0.0
    rate_L2 = 0.0
else:
    rate_max = np.polyfit(np.log(hs), np.log(errors_max), 1)[0]
    rate_L2 = np.polyfit(np.log(hs), np.log(errors_L2), 1)[0]

# --- 6. PLOTS ---
plt.figure(figsize=(10,6))
plt.plot(x_grid, T_exact, 'k-', label='Exact Analytical Solution', linewidth=2)
plt.plot(x_grid, T_corr, 'b--o', label='Corrected Numerical Solution')
plt.plot(x_grid, T_orig, 'r:s', label='Original Code 2 Solution')
plt.xlabel('Position $x$')
plt.ylabel('Temperature $T(x)$')
plt.title('Comparison of Temperature Profiles: Original vs Corrected')
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig('figures/temperature_comparison.png', dpi=300)
plt.savefig('figures/temperature_comparison.pdf')
plt.close()

# Error Plot
plt.figure(figsize=(10,6))
plt.plot(x_grid, err_orig, 'r:s', label='Error (Original)')
plt.plot(x_grid, err_corr, 'b--o', label='Error (Corrected)')
plt.xlabel('Position $x$')
plt.ylabel('Absolute Error $|T_{num} - T_{exact}|$')
plt.title('Absolute Error Comparison')
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig('figures/temperature_error_comparison.png', dpi=300)
plt.close()

# --- 7. SAVE RESULTS ---
results = {
    "parameters": {"k": k, "f": f, "bc": bc},
    "original": {
        "M": M_orig.tolist(),
        "s": s_orig.tolist(),
        "T": T_orig.tolist(),
        "residual": r_orig,
        "max_err": max_err_orig,
        "L2_err": L2_err_orig
    },
    "corrected": {
        "M": M_corr.tolist(),
        "s": s_corr.tolist(),
        "T": T_corr.tolist(),
        "residual": r_corr,
        "max_err": max_err_corr,
        "L2_err": L2_err_corr,
        "convergence_rate_max": rate_max,
        "convergence_rate_L2": rate_L2
    },
    "exact": T_exact.tolist()
}
with open("results/problem2_comparison.json", "w") as fp:
    json.dump(results, fp, indent=4)

with open("results/problem2_comparison.txt", "w") as fp:
    fp.write("Table: x | T_exact | T_orig | T_corr | err_orig | err_corr\n")
    for i in range(11):
        fp.write(f"{x_grid[i]:.2f} | {T_exact[i]:.6f} | {T_orig[i]:.6f} | {T_corr[i]:.6f} | {err_orig[i]:.6e} | {err_corr[i]:.6e}\n")

# --- 8. NOTEBOOK GENERATION ---
nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Problem 2 Correction\n",
    "This notebook implements the corrected finite-difference scheme for the steady-state heat equation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "k = {:.4f}\n".format(k),
    "f = {:.4f}\n".format(f),
    "h = 0.1\n",
    "N = 11\n",
    "x = np.linspace(0, 1, N)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analytical Solution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "T_exact = (f / (2*k)) * x * (1 - x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Corrected Finite-Difference Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "M_corr = np.zeros((N, N))\n",
    "s_corr = np.zeros(N)\n",
    "coef = 1/h**2\n",
    "for i in range(1, N-1):\n",
    "    M_corr[i, i-1] = -coef\n",
    "    M_corr[i, i] = 2*coef\n",
    "    M_corr[i, i+1] = -coef\n",
    "    s_corr[i] = f / k\n",
    "M_corr[0, 0] = 1\n",
    "s_corr[0] = 0\n",
    "M_corr[-1, -1] = 1\n",
    "s_corr[-1] = 0\n",
    "\n",
    "T_corr = np.linalg.solve(M_corr, s_corr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comparison Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8,5))\n",
    "plt.plot(x, T_exact, 'k-', label='Exact')\n",
    "plt.plot(x, T_corr, 'b--o', label='Corrected')\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
with open("notebooks/TC2_problem2_correction.ipynb", "w") as fp:
    json.dump(nb, fp, indent=2)

print("=== CORRECTION SCRIPT FINISHED ===")
print("Convergence rate (max error):", rate_max)
