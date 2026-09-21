import json
import numpy as np

with open('results/problem2_comparison.json', 'r') as f:
    comp = json.load(f)

with open('results/tc2_results.json', 'r') as f:
    res1 = json.load(f)

# Problem 1 matrix G
G = np.array(res1['problem1']['G'])
print("Matrix G:")
print("\\begin{pmatrix}")
for row in G:
    print(" & ".join([f"{v:.4f}" for v in row]) + " \\\\")
print("\\end{pmatrix}")

# Problem 1 vector x
x = np.array(res1['problem1']['x'])
print("\nVector x:")
print("\\begin{pmatrix}")
for v in x:
    print(f"{v:.4f} \\\\")
print("\\end{pmatrix}")

# Table 2: Comparison
x_grid = np.linspace(0, 1, 11)
T_exact = comp['exact']
T_orig = comp['original']['T']
T_corr = comp['corrected']['T']
err_orig = np.abs(np.array(T_orig) - np.array(T_exact))
err_corr = np.abs(np.array(T_corr) - np.array(T_exact))

print("\nTable 2 (Comparison):")
for i in range(11):
    print(f"{x_grid[i]:.1f} & {T_exact[i]:.4f} & {T_orig[i]:.4f} & {T_corr[i]:.4f} & {err_orig[i]:.4e} & {err_corr[i]:.4e} \\\\")

# Table 3: Convergence
print("\nTable 3 (Convergence):")
Ns = [11, 21, 41, 81]
hs = [1/(N-1) for N in Ns]
# Since we didn't save the convergence arrays in the json directly, we need to run it again quickly.
k = comp['parameters']['k']
f = comp['parameters']['f']
for N, h in zip(Ns, hs):
    xx = np.linspace(0, 1, N)
    MM = np.zeros((N, N))
    ss = np.zeros(N)
    cc = 1 / h**2
    for i in range(1, N - 1):
        MM[i, i-1] = -cc
        MM[i, i] = 2 * cc
        MM[i, i+1] = -cc
        ss[i] = f / k
    MM[0, 0] = -1/h
    MM[0, 1] = 1/h
    ss[0] = 0
    MM[-1, -1] = 1
    ss[-1] = 0
    TT = np.linalg.solve(MM, ss)
    TT_ex = (f / (2*k)) * (1 - xx**2)
    err = np.abs(TT - TT_ex)
    Linf = np.max(err)
    L2 = np.linalg.norm(err)*np.sqrt(h)
    print(f"{N} & {h:.4f} & {Linf:.4e} & {L2:.4e} \\\\")
