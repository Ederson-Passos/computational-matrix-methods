import numpy as np
import json
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

matricula = 31837
# --- RUN CODE 2 EXACTLY ---
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

M, s, k, f, bc = gerar_matriz_por_matricula(matricula)
T_num = np.linalg.solve(M, s)
x = np.linspace(0, 1, 11)

# Analytical solution of stated PDE: -k T'' = f  ==> T'' = -f/k
# T'(0) = 0 ==> T'(x) = -fx/k
# T(1) = 0 ==> T(1) = -f(1)^2/(2k) + C = 0 ==> C = f/(2k)
# T(x) = (f/(2k)) * (1 - x^2)
T_stated_analytic = (f / (2*k)) * (1 - x**2)

# Analytical solution of implemented PDE: T'' = (h^2/k) f
# But wait, does it have (h^2/k)*f on RHS and T'' on LHS?
# Actually, the finite difference LHS is T'' approx.
# So T'' = (h^2/k) f. Let's call F_impl = (h**2 / k) * f
h = 0.1
F_impl = (h**2 / k) * f
# T'(0) = F_impl * h ?
# The boundary condition in Code 2 is: (-T0 + T1)/h^2 = F_impl ==> (T1 - T0)/h = F_impl * h.
# This means T'(0) = F_impl * h.
# T''(x) = F_impl ==> T'(x) = F_impl * x + C1
# T'(0) = C1 = F_impl * h
# T'(x) = F_impl * (x + h)
# T(x) = F_impl * (0.5 * x**2 + h * x) + C2
# T(1) = 0 ==> F_impl * (0.5 + h) + C2 = 0 ==> C2 = -F_impl * (0.5 + h)
# T_impl(x) = F_impl * (0.5 * x**2 + h * x - 0.5 - h)

T_impl_analytic = F_impl * (0.5 * x**2 + h * x - 0.5 - h)

error_stated = np.abs(T_num - T_stated_analytic)
error_impl = np.abs(T_num - T_impl_analytic)

results = {
    "k": k,
    "f": f,
    "bc": bc,
    "h": h,
    "T_num": T_num.tolist(),
    "T_stated": T_stated_analytic.tolist(),
    "T_impl": T_impl_analytic.tolist(),
    "err_stated_max": np.max(error_stated),
    "err_impl_max": np.max(error_impl)
}

with open("results/problem2_audit.json", "w") as fp:
    json.dump(results, fp, indent=4)

print("err_stated_max =", np.max(error_stated))
print("err_impl_max =", np.max(error_impl))
