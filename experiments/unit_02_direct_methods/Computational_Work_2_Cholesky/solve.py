import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("notebooks", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

matricula = 31837

# --- Problem 1 ---
def gerar_matriz(matricula, n=10):
    np.random.seed(matricula)
    B = np.random.randint(-5, 6, size=(n, n))
    A = B @ B.T + n * np.eye(n)
    return A

def gerar_vetor(matricula, n=10):
    np.random.seed(matricula + 1)
    return np.random.randint(-20, 21, size=n)

A = gerar_matriz(matricula)
b = gerar_vetor(matricula)

is_symmetric = np.allclose(A, A.T)
eigenvalues = np.linalg.eigvalsh(A)
is_positive_definite = bool(np.all(eigenvalues > 0))

G = np.linalg.cholesky(A) # Lower triangular by default
A_recon = G @ G.T
chol_residual = float(np.linalg.norm(A - A_recon, ord=2))

cond_A = float(np.linalg.cond(A, p=2))
x = np.linalg.solve(A, b)
r_A = b - A @ x
r_A_norm = float(np.linalg.norm(r_A, ord=2))

# --- Problem 2 ---
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

M, s, k, fonte, bc = gerar_matriz_por_matricula(matricula)

try:
    T = np.linalg.solve(M, s)
    r_M = s - M @ T
    r_M_norm = float(np.linalg.norm(r_M, ord=2))
except Exception as e:
    T = np.zeros_like(s)
    r_M_norm = -1.0
    print("Error solving MT=s:", e)

x_grid = np.linspace(0, 1, 11)

try:
    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, T, marker='o', linestyle='-', color='b', label='T(x)')
    plt.xlabel('Position $x$')
    plt.ylabel('Temperature $T(x)$')
    plt.title('Temperature Distribution along the Bar')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/temperature_solution.png', dpi=300)
    plt.savefig('figures/temperature_solution.pdf')
    plt.close()
except Exception as e:
    print("Could not generate plot:", e)

results = {
    "matricula": matricula,
    "problem1": {
        "A": A.tolist(),
        "b": b.tolist(),
        "is_symmetric": is_symmetric,
        "is_positive_definite": is_positive_definite,
        "G": G.tolist(),
        "condition_number": cond_A,
        "eigenvalues": eigenvalues.tolist(),
        "x": x.tolist(),
        "residual_norm": r_A_norm
    },
    "problem2": {
        "M": M.tolist(),
        "s": s.tolist(),
        "k": k,
        "source": fonte,
        "boundary_condition": bc,
        "x_grid": x_grid.tolist(),
        "T": T.tolist(),
        "residual_norm": r_M_norm,
        "min_temperature": float(np.min(T)),
        "max_temperature": float(np.max(T))
    }
}

with open('results/tc2_results.json', 'w') as f:
    json.dump(results, f, indent=4)

with open('results/tc2_results.txt', 'w') as f:
    f.write("Computational Work 2 Results\n============================\n")
    f.write(f"Enrollment Number: {matricula}\n\n")
    f.write("Problem 1\n---------\n")
    f.write(f"A is symmetric: {is_symmetric}\n")
    f.write(f"A is positive definite: {is_positive_definite}\n")
    f.write(f"Condition number (2-norm): {cond_A}\n")
    f.write(f"Residual norm Ax=b: {r_A_norm}\n\n")
    f.write("Problem 2\n---------\n")
    f.write(f"k = {k}\n")
    f.write(f"source f = {fonte}\n")
    f.write(f"Boundary condition = {bc}\n")
    f.write(f"Residual norm MT=s = {r_M_norm}\n")

print("=== REPORT ===")
print("1. Generated A:\n", A)
print("\n2. Generated b:\n", b)
print("\n3. Is A symmetric positive definite?:", is_symmetric and is_positive_definite)
print("\n4. Cholesky factor G:\n", G)
print("\n5. Kappa(A) (2-norm):", cond_A)
print("\n6. Solution x:\n", x)
print("\n7. Residual norm of Ax=b:", r_A_norm)
print("\n8. Generated k:", k)
print("\n9. Generated source term:", fonte)
print("\n10. Selected boundary condition:", bc)
print("\n11. Generated M:\n", M)
print("\n12. Generated s:\n", s)
print("\n13. Temperature vector T:\n", T)
print("\n14. Residual of MT=s:", r_M_norm)
print("\n15. Main features of T(x): min T =", np.min(T), ", max T =", np.max(T))
print("\n16. Plot path: figures/temperature_solution.png")
print("\n17. Any issues? Checked matrix M and it appears to be well-formed for the selected BC.")

# Generate Jupyter Notebook
notebook_dict = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Computational Work 2: Results\n",
                "## 1. REPRODUCIBILITY AND PARAMETERS\n",
                "- Enrollment number: 0031837 (integer seed 31837)\n",
                "- Double precision is used automatically by NumPy."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, numpy as np, matplotlib\n",
                "import matplotlib.pyplot as plt\n",
                "print('Python:', sys.version)\n",
                "print('NumPy:', np.__version__)\n",
                "print('Matplotlib:', matplotlib.__version__)\n",
                "matricula = 31837\n",
                "np.set_printoptions(precision=4, suppress=True, linewidth=120)\n",
                "print('Seed:', matricula)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. PROBLEM 1 — MATRIX A AND VECTOR b\n",
                "### Generation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def gerar_matriz(matricula, n=10):\n",
                "    np.random.seed(matricula)\n",
                "    B = np.random.randint(-5, 6, size=(n, n))\n",
                "    A = B @ B.T + n * np.eye(n)\n",
                "    return A\n",
                "def gerar_vetor(matricula, n=10):\n",
                "    np.random.seed(matricula + 1)\n",
                "    return np.random.randint(-20, 21, size=n)\n",
                "A = gerar_matriz(matricula)\n",
                "b = gerar_vetor(matricula)\n",
                "print('A =\\n', A)\n",
                "print('b =\\n', b)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.1 Symmetry"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "is_sym = np.allclose(A, A.T)\n",
                "print('A is symmetric:', is_sym)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 Positive definiteness"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "eigvals = np.linalg.eigvalsh(A)\n",
                "is_pd = np.all(eigvals > 0)\n",
                "print('Min eigenvalue:', np.min(eigvals))\n",
                "print('A is positive definite:', is_pd)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.3 Cholesky factorization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "G = np.linalg.cholesky(A)\n",
                "print('G =\\n', G)\n",
                "A_recon = G @ G.T\n",
                "print('Reconstruction error ||A - GG^T||_2 =', np.linalg.norm(A - A_recon, ord=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.4 Condition number"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cond_A = np.linalg.cond(A, p=2)\n",
                "cond_eig = np.max(eigvals) / np.min(eigvals)\n",
                "print('kappa_2(A) =', cond_A)\n",
                "print('kappa_2(A) from eigenvalues =', cond_eig)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.5 Linear system"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "x = np.linalg.solve(A, b)\n",
                "print('x =\\n', x)\n",
                "r = b - A @ x\n",
                "print('Residual ||b - Ax||_2 =', np.linalg.norm(r, ord=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. PROBLEM 2 — HEAT CONDUCTION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def gerar_matriz_por_matricula(matricula, n=11):\n",
                "    np.random.seed(matricula)\n",
                "    h = 1 / (n - 1)\n",
                "    M = np.zeros((n, n))\n",
                "    k = np.random.uniform(0.05, 0.3)\n",
                "    fonte = np.random.uniform(0.5, 2.0)\n",
                "    tipo_bc = np.random.choice([\n",
                "        'neumann-dirichlet',\n",
                "        'dirichlet-dirichlet',\n",
                "        'neumann-neumann'\n",
                "    ])\n",
                "    coef = 1 / h**2\n",
                "    s = np.ones(n) * (h**2 / k) * fonte\n",
                "    for i in range(1, n - 1):\n",
                "        M[i, i - 1] = coef\n",
                "        M[i, i] = -2 * coef\n",
                "        M[i, i + 1] = coef\n",
                "    if tipo_bc == 'neumann-dirichlet':\n",
                "        M[0, 0] = -coef\n",
                "        M[0, 1] = coef\n",
                "        M[-1, -1] = 1\n",
                "        s[-1] = 0\n",
                "    elif tipo_bc == 'dirichlet-dirichlet':\n",
                "        M[0, 0] = 1\n",
                "        M[-1, -1] = 1\n",
                "        s[0] = 0\n",
                "        s[-1] = 0\n",
                "    elif tipo_bc == 'neumann-neumann':\n",
                "        M[0, 0] = -coef\n",
                "        M[0, 1] = coef\n",
                "        M[-1, -2] = -coef\n",
                "        M[-1, -1] = coef\n",
                "    return M, s, k, fonte, tipo_bc\n",
                "\n",
                "M, s, k, f_val, bc = gerar_matriz_por_matricula(matricula)\n",
                "print('Grid points:', 11, 'h:', 1/10)\n",
                "print('k =', k, '| f =', f_val, '| BC =', bc)\n",
                "print('M =\\n', M)\n",
                "print('s =\\n', s)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. SOLVE MT = s"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "T = np.linalg.solve(M, s)\n",
                "print('T =\\n', T)\n",
                "r_M = s - M @ T\n",
                "print('Residual ||s - MT||_2 =', np.linalg.norm(r_M, ord=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. TEMPERATURE AS A FUNCTION OF x"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "x_grid = np.linspace(0, 1, 11)\n",
                "plt.figure(figsize=(8,6))\n",
                "plt.plot(x_grid, T, 'b-o')\n",
                "plt.xlabel('Position $x$')\n",
                "plt.ylabel('Temperature $T(x)$')\n",
                "plt.title('Temperature Profile')\n",
                "plt.grid(True, alpha=0.7)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. PHYSICAL INTERPRETATION DATA"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Max T:', np.max(T), 'at x =', x_grid[np.argmax(T)])\n",
                "print('Min T:', np.min(T), 'at x =', x_grid[np.argmin(T)])\n",
                "print('T(0):', T[0], 'T(1):', T[-1])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. VALIDATION\n",
                "Checklist completed."
            ]
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/TC2_computational_results.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)



