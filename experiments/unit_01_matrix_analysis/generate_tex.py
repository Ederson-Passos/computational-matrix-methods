import numpy as np

# Same seed and matrix from Item (a)
np.random.seed(42)
A = np.random.choice([-1, 0, 1], size=(8, 16))

# Full SVD calculation
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Construction of the Sigma matrix (8x16) from the 1D array of singular values
Sigma = np.zeros((8, 16))
np.fill_diagonal(Sigma, S)

# Recovering the V matrix from Vt
V = Vt.T

# --- Helper function to generate LaTeX code ---
def matrix_to_latex(matrix, name, precision=4):
    """Converts a numpy array to a LaTeX pmatrix string."""
    latex_str = f"\\textbf{{Matrix }} ${name} \\in \\mathbb{{R}}^{{{matrix.shape[0]} \\times {matrix.shape[1]}}}$: \n"
    latex_str += "\\begin{equation*}\n"
    # If the matrix is too large (like 16x16), we use resizebox to fit on the page
    if matrix.shape[1] > 8:
        latex_str += "\\resizebox{\\textwidth}{!}{$\n"
    
    latex_str += "\\begin{bmatrix}\n"
    for row in matrix:
        # Format each number with the required precision
        formatted_row = [f"{val:.{precision}f}" for val in row]
        latex_str += " & ".join(formatted_row) + " \\\\\n"
    latex_str += "\\end{bmatrix}\n"
    
    if matrix.shape[1] > 8:
        latex_str += "$}\n"
    latex_str += "\\end{equation*}\n"
    return latex_str

# Outputs to terminal and writes to LaTeX file
output_lines = [
    "% --- Lines for Overleaf ---",
    matrix_to_latex(U, "U", precision=6),
    matrix_to_latex(Sigma, "\\Sigma", precision=6),
    matrix_to_latex(V, "V", precision=6)
]

with open("question1c_answer.tex", "w") as f:
    for line in output_lines:
        f.write(line + "\n")
