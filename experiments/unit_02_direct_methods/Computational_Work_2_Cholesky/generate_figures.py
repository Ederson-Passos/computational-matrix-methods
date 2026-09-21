import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os

os.makedirs('figures', exist_ok=True)

# Tufte/TEC style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "cm"
})

# Load Data
with open('results/tc2_results.json', 'r') as f:
    tc2 = json.load(f)

with open('results/problem2_comparison.json', 'r') as f:
    comp = json.load(f)

# --- Figure 1: Problem 1 Matrix Structure ---
A = np.array(tc2['problem1']['A'])
G = np.array(tc2['problem1']['G'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

cmap = 'RdBu_r'

vmax_A = np.max(np.abs(A))
im1 = ax1.imshow(A, cmap=cmap, vmin=-vmax_A, vmax=vmax_A)
ax1.set_title("Generated SPD matrix $A$")
ax1.set_xticks([])
ax1.set_yticks([])
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Mask upper triangle of G for visual clarity
G_masked = np.copy(G)
G_masked[np.triu_indices_from(G, k=1)] = np.nan
cmap_G = plt.cm.RdBu_r
cmap_G.set_bad('white', 1.)

vmax_G = np.nanmax(np.abs(G_masked))
im2 = ax2.imshow(G_masked, cmap=cmap_G, vmin=-vmax_G, vmax=vmax_G)
ax2.set_title("Cholesky factor $G$")
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

fig.savefig('figures/problem1_matrix_cholesky.pdf', bbox_inches='tight')
fig.savefig('figures/problem1_matrix_cholesky.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figure 2: Problem 2 Physical Configuration ---
fig, ax = plt.subplots(figsize=(8, 2))
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')

# Bar
bar = Rectangle((0, 0), 1, 1, facecolor='#eaeaea', edgecolor='black', linewidth=1)
ax.add_patch(bar)

# Source f (Distributed volumetric generation)
for i in np.linspace(0.05, 0.95, 19):
    ax.text(i, 0.5, '+', ha='center', va='center', color='#cc0000', fontsize=12)
f_val = comp['parameters']['f']
ax.text(0.5, 0.8, f"Uniform source $f = {f_val:.4f}$", ha='center', va='center', color='#cc0000', fontsize=11)

# Labels
k_val = comp['parameters']['k']
ax.text(0.5, 1.35, f"Thermal conductivity: $k = {k_val:.4f}$", ha='center', va='center', fontsize=11)
ax.text(0.5, -0.2, "$x$", ha='center', va='center', fontsize=11)
ax.text(0.0, -0.2, "$x=0$", ha='center', va='center', fontsize=11)
ax.text(1.0, -0.2, "$x=1$", ha='center', va='center', fontsize=11)

# BC Left
ax.plot([0, 0], [0, 1], color='blue', linewidth=3)
ax.text(-0.05, 0.5, "$T'(0)=0$\n(Insulated)", ha='right', va='center', fontsize=11, color='blue')

# BC Right
ax.plot([1, 1], [0, 1], color='green', linewidth=3)
ax.text(1.05, 0.5, "$T(1)=0$\n(Prescribed temperature)", ha='left', va='center', fontsize=11, color='green')

fig.savefig('figures/problem2_physical_configuration.pdf', bbox_inches='tight')
fig.savefig('figures/problem2_physical_configuration.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figure 3: Temperature Comparison ---
x_grid = np.linspace(0, 1, 11)
x_dense = np.linspace(0, 1, 500)
T_exact_dense = (comp['parameters']['f'] / (2 * comp['parameters']['k'])) * (1 - x_dense**2)
T_orig = comp['original']['T']
T_corr = comp['corrected']['T']

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_dense, T_exact_dense, '-', color='black', label='Exact Analytical', linewidth=1.5)
ax.plot(x_grid, T_orig, 's', color='#d95f02', label='Original Formulation', markersize=5, fillstyle='none', linestyle='none')
ax.plot(x_grid, T_corr, 'o', color='#1b9e77', label='Corrected Formulation', markersize=5, fillstyle='none', linestyle='none')
ax.set_xlabel('Position $x$')
ax.set_ylabel('Temperature $T(x)$')
ax.legend(frameon=False)
ax.grid(True, linestyle=':')

fig.savefig('figures/temperature_comparison.pdf', bbox_inches='tight')
fig.savefig('figures/temperature_comparison.png', bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figure 4: Convergence ---
k = comp['parameters']['k']
f = comp['parameters']['f']

Ns = [11, 21, 41, 81]
hs = [1/(N-1) for N in Ns]

errors_Linf = []
errors_L2 = []

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
    errors_Linf.append(np.max(err))
    errors_L2.append(np.linalg.norm(err)*np.sqrt(h))

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(Ns, errors_Linf, 'o-', color='#1b9e77', label='$L_\infty$ Error', fillstyle='none')
ax.loglog(Ns, errors_L2, 's-', color='#7570b3', label='Discrete $L_2$ Error', fillstyle='none')

# Reference line O(h) which corresponds to O(N^-1)
Ns_ref = np.array([Ns[0], Ns[-1]])
ref_start = errors_Linf[0] * 1.5
ax.loglog(Ns_ref, ref_start * (Ns_ref / Ns[0])**(-1), 'k--', label='First-order reference slope', linewidth=1)

ax.set_xticks(Ns)
ax.set_xticklabels([str(N) for N in Ns])
ax.set_xlabel('Number of grid points $N$')
ax.set_ylabel('Error')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', which='both')

fig.savefig('figures/problem2_convergence.pdf', bbox_inches='tight')
fig.savefig('figures/problem2_convergence.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Generated all 4 figures.")
