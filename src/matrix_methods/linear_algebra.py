"""Routines for Numerical Linear Algebra implemented from first principles.

This module implements the fundamental algorithms for QR factorization via
Modified Gram-Schmidt (MGS), back substitution for upper-triangular systems,
linear least-squares solutions, and numerical verification metrics.

References:
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
      The Johns Hopkins University Press.
    - Trefethen, L. N., & Bau, D., III (1997). Numerical Linear Algebra. SIAM.
    - Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence


class RankDeficiencyError(Exception):
    """Exception raised when a matrix is detected to be numerically rank-deficient."""

    pass


# =============================================================================
# Helper & Conversion Functions
# =============================================================================

def to_float_matrix(M: Sequence[Sequence[Any]]) -> list[list[float]]:
    """Converts a nested sequence (e.g. np.ndarray or list of lists) to a list of lists of floats."""
    return [[float(val) for val in row] for row in M]


def to_float_vector(v: Sequence[Any]) -> list[float]:
    """Converts a sequence (e.g. 1D np.ndarray or list) to a list of floats."""
    return [float(val) for val in v]


# =============================================================================
# Vector Operations
# =============================================================================

def vector_dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Computes the Euclidean inner product (dot product) of two vectors.

    .. math::
        \\langle \\mathbf{u}, \\mathbf{v} \\rangle = \\sum_{i=0}^{n-1} u_i v_i

    Args:
        u: First vector of dimension n.
        v: Second vector of dimension n.

    Returns:
        The scalar inner product.

    Raises:
        ValueError: If u and v have different dimensions.
    """
    if len(u) != len(v):
        raise ValueError(f"Incompatible vector dimensions for dot product: {len(u)} != {len(v)}")
    return sum(ui * vi for ui, vi in zip(u, v))


def vector_norm(u: Sequence[float]) -> float:
    """Computes the standard Euclidean norm (L2 norm) of a vector.

    .. math::
        \\|\\mathbf{u}\\|_2 = \\sqrt{\\sum_{i=0}^{n-1} u_i^2}

    Args:
        u: Vector of dimension n.

    Returns:
        The Euclidean norm as a non-negative float.
    """
    return math.sqrt(sum(ui * ui for ui in u))


def vector_add(u: Sequence[float], v: Sequence[float]) -> list[float]:
    """Computes the element-wise sum of two vectors u + v."""
    if len(u) != len(v):
        raise ValueError(f"Incompatible vector dimensions for addition: {len(u)} != {len(v)}")
    return [ui + vi for ui, vi in zip(u, v)]


def vector_sub(u: Sequence[float], v: Sequence[float]) -> list[float]:
    """Computes the element-wise difference of two vectors u - v."""
    if len(u) != len(v):
        raise ValueError(f"Incompatible vector dimensions for subtraction: {len(u)} != {len(v)}")
    return [ui - vi for ui, vi in zip(u, v)]


def vector_scale(u: Sequence[float], scalar: float) -> list[float]:
    """Multiplies vector u by a scalar: scalar * u."""
    return [scalar * ui for ui in u]


# =============================================================================
# Matrix Operations
# =============================================================================

def matrix_shape(A: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Returns the dimension (m, n) of matrix A."""
    m = len(A)
    if m == 0:
        return 0, 0
    n = len(A[0])
    return m, n


def get_column(A: Sequence[Sequence[float]], col_idx: int) -> list[float]:
    """Extracts column col_idx from matrix A as a vector of length m."""
    return [row[col_idx] for row in A]


def set_column(A: list[list[float]], col_idx: int, col_vec: Sequence[float]) -> None:
    """Sets the values of column col_idx in matrix A from col_vec in place."""
    for i, val in enumerate(col_vec):
        A[i][col_idx] = val


def matrix_transpose(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Computes the transpose A^T of matrix A.

    For A in R^{m x n}, the transpose A^T in R^{n x m} satisfies:
    (A^T)_{ij} = A_{ji}.
    """
    m, n = matrix_shape(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matrix_vector_mul(A: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    """Computes matrix-vector multiplication y = A * v.

    For A in R^{m x n} and v in R^n:
    y_i = sum_{j=0}^{n-1} A_{ij} v_j, for i = 0, ..., m-1.
    """
    m, n = matrix_shape(A)
    if len(v) != n:
        raise ValueError(f"Incompatible dimensions for A * v: matrix has {n} cols, vector has {len(v)} elements")
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]


def matrix_matrix_mul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    """Computes matrix-matrix product C = A * B.

    For A in R^{m x k} and B in R^{k x n}:
    C_{ij} = sum_{p=0}^{k-1} A_{ip} B_{pj}.
    """
    m, k_a = matrix_shape(A)
    k_b, n = matrix_shape(B)
    if k_a != k_b:
        raise ValueError(f"Incompatible matrix dimensions for A * B: {m}x{k_a} and {k_b}x{n}")
    return [[sum(A[i][p] * B[p][j] for p in range(k_a)) for j in range(n)] for i in range(m)]


def matrix_sub(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    """Computes element-wise matrix subtraction C = A - B."""
    m_a, n_a = matrix_shape(A)
    m_b, n_b = matrix_shape(B)
    if (m_a, n_a) != (m_b, n_b):
        raise ValueError(f"Incompatible matrix dimensions for subtraction: ({m_a}, {n_a}) != ({m_b}, {n_b})")
    return [[A[i][j] - B[i][j] for j in range(n_a)] for i in range(m_a)]


def frobenius_norm(A: Sequence[Sequence[float]]) -> float:
    """Computes the Frobenius norm of a matrix.

    .. math::
        \\|A\\|_F = \\sqrt{\\sum_{i=0}^{m-1} \\sum_{j=0}^{n-1} A_{ij}^2}
    """
    return math.sqrt(sum(val * val for row in A for val in row))


def identity_matrix(n: int) -> list[list[float]]:
    """Constructs the n x n identity matrix I_n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# =============================================================================
# Modified Gram-Schmidt QR Factorization
# =============================================================================

def qr_mgs(
    A: Sequence[Sequence[float]],
    tol: float | None = None,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
    """Computes the reduced QR factorization using Modified Gram-Schmidt (MGS).

    Given A in R^{m x n} with m >= n, finds:
        A = Q * R
    where Q in R^{m x n} has orthonormal columns (Q^T Q = I_n) and
    R in R^{n x n} is upper triangular with non-negative diagonal elements.

    Algorithm Procedure:
        1. Extract the column vectors of the original matrix: v_j^(0) = a_j.
        2. Compute the necessary inner products: r_{ij} = <q_i, v_j^(i-1)>.
        3. Sequentially subtract projections onto previous orthonormal vectors:
           v_j^(i) = v_j^(i-1) - r_{ij} q_i.
        4. Compute Euclidean norm of residual vector: r_{jj} = ||v_j^(j-1)||_2.
        5. Check numerical rank deficiency against tolerance: r_{jj} <= tol.
        6. Normalize the residual vector to form the new column of Q:
           q_j = v_j^(j-1) / r_{jj}.
        7. Store projection coefficients and norms in upper-triangular matrix R.

    Numerical Rank Deficiency & Limitations:
        A matrix is flagged as numerically rank-deficient if any diagonal entry
        r_{jj} <= tol. The default tolerance is scale-aware:
            tol = max(m, n) * eps_mach * ||A||_F.
        Note: Modified Gram-Schmidt without column pivoting does NOT constitute
        a universally reliable rank-revealing test for arbitrary ill-conditioned
        matrices, because rounding errors may obscure rank deficiency or lead
        to loss of orthogonality (e.g. in Kahan matrices or clustered singular values).
        For guaranteed rank-revealing factorization, SVD or QR with column pivoting
        (RRQR) should be used.

    Args:
        A: Input rectangular matrix of shape (m, n) with m >= n.
        tol: Tolerance for detecting rank deficiency. If None, uses
             max(m, n) * sys.float_info.epsilon * ||A||_F.

    Returns:
        A tuple (Q, R, diag_info) where:
            Q: m x n matrix with orthonormal columns.
            R: n x n upper-triangular matrix.
            diag_info: Dictionary containing rank status, tolerance, condition,
                       and diagonal entries.

    Raises:
        ValueError: If m < n or matrix has invalid dimensions.
    """
    A_mat = to_float_matrix(A)
    m, n = matrix_shape(A_mat)

    if m < n:
        raise ValueError(f"QR factorization requires m >= n, but got shape ({m}, {n}).")
    if n == 0:
        raise ValueError("Cannot perform QR factorization on an empty matrix.")

    frob_A = frobenius_norm(A_mat)
    eps_mach = sys.float_info.epsilon

    if tol is None:
        tol = max(m, n) * eps_mach * frob_A

    # Initialize R as an n x n zero matrix
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    # Initialize Q columns storage
    q_cols: list[list[float]] = []

    # Initialize working vectors as copies of columns of A
    v_cols: list[list[float]] = [get_column(A_mat, j) for j in range(n)]

    is_full_rank = True
    rank_deficiency_col: int | None = None
    rank_deficiency_reason: str | None = None

    for j in range(n):
        v_j = v_cols[j]

        # Step 2 & 3: Project v_j against all previously computed orthonormal vectors q_i
        for i in range(j):
            q_i = q_cols[i]
            r_ij = vector_dot(q_i, v_j)
            R[i][j] = r_ij
            # Modified Gram-Schmidt update: subtract projection immediately
            v_j = vector_sub(v_j, vector_scale(q_i, r_ij))

        # Step 4: Compute Euclidean norm of residual vector
        r_jj = vector_norm(v_j)
        R[j][j] = r_jj

        # Step 5: Check numerical rank deficiency
        if r_jj <= tol:
            is_full_rank = False
            if rank_deficiency_col is None:
                rank_deficiency_col = j
                rank_deficiency_reason = (
                    f"Diagonal r_{{{j},{j}}} = {r_jj:.6e} <= tol = {tol:.6e}. "
                    f"Possível dependência linear detectada na coluna {j}."
                )
            # Create a zero vector to avoid division by zero
            q_j = [0.0] * m
        else:
            # Step 6: Normalize to construct new orthonormal column
            q_j = vector_scale(v_j, 1.0 / r_jj)

        q_cols.append(q_j)

    # Reconstruct Q as an m x n matrix from column vectors
    Q = [[q_cols[j][i] for j in range(n)] for i in range(m)]

    diag_info: dict[str, Any] = {
        "is_full_rank": is_full_rank,
        "tolerance": tol,
        "frobenius_norm_A": frob_A,
        "diagonal_entries": [R[i][i] for i in range(n)],
        "rank_deficiency_col": rank_deficiency_col,
        "rank_deficiency_reason": rank_deficiency_reason,
    }

    return Q, R, diag_info


# =============================================================================
# Back Substitution for Upper-Triangular Systems
# =============================================================================

def back_substitution(
    R: Sequence[Sequence[float]],
    d: Sequence[float],
    tol: float = 1e-15,
) -> list[float]:
    """Solves the upper-triangular linear system R * x = d by backward substitution.

    Algorithm:
        x_i = \\frac{d_i - \\sum_{j=i+1}^{n-1} R_{ij} x_j}{R_{ii}},
        for i = n-1, n-2, ..., 0.

    Args:
        R: Upper-triangular matrix of shape (n, n).
        d: Right-hand side vector of length n.
        tol: Minimum acceptable absolute value for diagonal pivots.

    Returns:
        The solution vector x of length n.

    Raises:
        RankDeficiencyError: If a diagonal element |R_{ii}| <= tol.
        ValueError: If dimensions of R and d are incompatible.
    """
    R_mat = to_float_matrix(R)
    d_vec = to_float_vector(d)
    n, cols = matrix_shape(R_mat)

    if n != cols:
        raise ValueError(f"Matrix R must be square, got shape ({n}, {cols}).")
    if len(d_vec) != n:
        raise ValueError(f"Dimension mismatch: R is {n}x{n}, but d has length {len(d_vec)}.")

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        pivot = R_mat[i][i]
        if abs(pivot) <= tol:
            raise RankDeficiencyError(
                f"Pivô nulo ou numericamente desprezível encontrado em R[{i}][{i}] = {pivot:.6e} "
                f"(limiar: {tol:.6e}). Sistema singular ou mal condicionado."
            )
        partial_sum = sum(R_mat[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (d_vec[i] - partial_sum) / pivot

    return x


# =============================================================================
# Linear Least-Squares Solver via QR
# =============================================================================

def solve_least_squares_qr(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    tol: float | None = None,
) -> tuple[list[float], list[list[float]], list[list[float]], dict[str, Any]]:
    """Computes the unique least-squares solution to A * x \\approx b using QR factorization.

    Mathematical Justification:
        Consider the overdetermined linear system A x \\approx b, where A in R^{m x n}
        with m >= n has full column rank (rank(A) = n).
        We seek the vector x* that minimizes the squared Euclidean residual:
            ||A x - b||_2^2.

        Using the reduced QR factorization A = Q R (where Q in R^{m x n} has
        orthonormal columns and R in R^{n x n} is upper triangular and nonsingular):
            ||A x - b||_2^2 = ||Q R x - b||_2^2.

        Decomposing b along the range space Col(Q) = Col(A) and its orthogonal complement:
            b = Q (Q^T b) + (I - Q Q^T) b.
        Because Q (Q^T b) is in Col(Q) and (I - Q Q^T) b is orthogonal to Col(Q):
            A x - b = Q (R x - Q^T b) - (I - Q Q^T) b.
        By the Pythagorean theorem for orthogonal subspaces:
            ||A x - b||_2^2 = ||R x - Q^T b||_2^2 + ||(I - Q Q^T) b||_2^2.

        The second term ||(I - Q Q^T) b||_2^2 is constant and independent of x.
        The first term ||R x - Q^T b||_2^2 >= 0 attains its global minimum of 0
        if and only if:
            R x = Q^T b.

        Since rank(A) = n, the diagonal entries of R are strictly non-zero, making R
        invertible. Thus, back substitution yields the unique minimizer x* = R^{-1} Q^T b.

    Args:
        A: Matrix of shape (m, n) with m >= n.
        b: Right-hand side vector of length m.
        tol: Tolerance for detecting rank deficiency in QR.

    Returns:
        A tuple (x_ls, Q, R, diag_info).

    Raises:
        RankDeficiencyError: If A is detected to be rank-deficient.
    """
    A_mat = to_float_matrix(A)
    b_vec = to_float_vector(b)
    m, n = matrix_shape(A_mat)

    if len(b_vec) != m:
        raise ValueError(f"Incompatible dimensions: A has {m} rows, but b has length {len(b_vec)}.")

    # Step 1: Compute reduced QR factorization
    Q, R, diag_info = qr_mgs(A_mat, tol=tol)

    if not diag_info["is_full_rank"]:
        reason = diag_info.get("rank_deficiency_reason", "Possível deficiência de posto detectada.")
        raise RankDeficiencyError(
            f"Falha na resolução de mínimos quadrados: a matriz A não possui posto completo.\n"
            f"Detalhes: {reason}\n"
            f"O algoritmo MGS foi interrompido para evitar divisão por pivô numérico quase-nulo."
        )

    # Step 2: Explicit transpose Q^T
    Q_T = matrix_transpose(Q)

    # Step 3: Compute d = Q^T * b
    d = matrix_vector_mul(Q_T, b_vec)

    # Step 4: Back substitution R * x = d
    x_ls = back_substitution(R, d, tol=diag_info["tolerance"])

    return x_ls, Q, R, diag_info


# =============================================================================
# Numerical Verification Indicators
# =============================================================================

def compute_qr_indicators(
    A: Sequence[Sequence[float]],
    Q: Sequence[Sequence[float]],
    R: Sequence[Sequence[float]],
    x: Sequence[float],
    b: Sequence[float],
) -> dict[str, Any]:
    """Evaluates the four core numerical verification indicators for QR least squares.

    1. Relative QR Reconstruction Error:
       .. math::
           \\eta_{\\mathrm{QR}} = \\frac{\\|A - QR\\|_F}{\\|A\\|_F}
       Measures how faithfully the product Q * R reproduces the original matrix A.
       Reflects floating-point roundoff in the factorization and should be near
       machine epsilon (e.g. 10^{-16} to 10^{-15}).

    2. Orthogonality Error:
       .. math::
           \\eta_Q = \\|Q^T Q - I_n\\|_F
       Measures the departure of Q from perfect orthonormality. For MGS, this error
       grows proportionally to kappa_2(A) * eps_mach.

    3. System Residual Norm:
       .. math::
           \\rho = \\|A\\mathbf{x}^* - \\mathbf{b}\\|_2
       The Euclidean norm of the residual vector r = A x* - b. In an overdetermined
       incompatible system, rho > 0 is expected and represents the geometric distance
       from b to Col(A). A non-zero residual does NOT indicate error; it is the fundamental
       least-squares quantity being minimized.

    4. Least-Squares Optimality / Normal Condition Verification:
       .. math::
           \\eta_{\\mathrm{LS}} = \\|A^T (A\\mathbf{x}^* - \\mathbf{b})\\|_2
       Measures the orthogonality of the residual vector to the column space Col(A),
       verifying the first-order stationarity condition \\nabla ||A x - b||_2^2 = 2 A^T r = 0.
       This indicator must be close to zero (on the scale of machine precision), confirming
       that x* is indeed the true minimizer.

    Args:
        A: Original matrix m x n.
        Q: Orthonormal matrix m x n.
        R: Upper-triangular matrix n x n.
        x: Computed least-squares solution vector of length n.
        b: Right-hand side vector of length m.

    Returns:
        A dictionary with all indicators and diagnostic details.
    """
    A_mat = to_float_matrix(A)
    Q_mat = to_float_matrix(Q)
    R_mat = to_float_matrix(R)
    x_vec = to_float_vector(x)
    b_vec = to_float_vector(b)

    m, n = matrix_shape(A_mat)

    # 1. Relative QR reconstruction error
    QR = matrix_matrix_mul(Q_mat, R_mat)
    diff_A_QR = matrix_sub(A_mat, QR)
    frob_diff = frobenius_norm(diff_A_QR)
    frob_A = frobenius_norm(A_mat)
    eta_QR = frob_diff / frob_A if frob_A > 0.0 else frob_diff

    # 2. Orthogonality error
    Q_T = matrix_transpose(Q_mat)
    QT_Q = matrix_matrix_mul(Q_T, Q_mat)
    I_n = identity_matrix(n)
    diff_ortho = matrix_sub(QT_Q, I_n)
    eta_Q = frobenius_norm(diff_ortho)

    # 3. Residual vector and Euclidean norm
    Ax = matrix_vector_mul(A_mat, x_vec)
    residual_vector = vector_sub(Ax, b_vec)
    rho = vector_norm(residual_vector)

    # 4. Least-squares orthogonality verification: ||A^T * r||_2
    A_T = matrix_transpose(A_mat)
    AT_r = matrix_vector_mul(A_T, residual_vector)
    eta_LS = vector_norm(AT_r)

    return {
        "eta_QR": eta_QR,
        "eta_Q": eta_Q,
        "rho": rho,
        "eta_LS": eta_LS,
        "residual_vector": residual_vector,
        "frob_A": frob_A,
    }
