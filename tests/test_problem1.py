"""Automated unit test suite for Problem 1 (QR Factorization and Least Squares).

All tests use purely synthetic, deterministic data and are strictly independent
of any student enrollment number (matrícula).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# Ensure src is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from matrix_methods.linear_algebra import (
    RankDeficiencyError,
    back_substitution,
    compute_qr_indicators,
    frobenius_norm,
    identity_matrix,
    matrix_matrix_mul,
    matrix_sub,
    matrix_transpose,
    matrix_vector_mul,
    qr_mgs,
    solve_least_squares_qr,
    vector_dot,
    vector_norm,
)
from matrix_methods.problem1 import (
    build_problem1_data,
    run_problem1,
    validate_and_process_matricula,
)


class TestLinearAlgebraCore(unittest.TestCase):
    """Tests for vector, matrix primitives, and numerical linear algebra operations."""

    def test_vector_operations(self):
        u = [1.0, 2.0, -3.0]
        v = [4.0, -1.0, 2.0]
        # Dot product: 1*4 + 2*(-1) + (-3)*2 = 4 - 2 - 6 = -4
        self.assertAlmostEqual(vector_dot(u, v), -4.0, places=14)
        # Norm: sqrt(1 + 4 + 9) = sqrt(14)
        self.assertAlmostEqual(vector_norm(u), math.sqrt(14.0), places=14)

    def test_matrix_vector_and_matrix_mul(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        x = [5.0, 6.0]
        # Ax = [17, 39]
        Ax = matrix_vector_mul(A, x)
        self.assertEqual(Ax, [17.0, 39.0])

        B = [[2.0, 0.0], [1.0, 2.0]]
        # AB = [[4, 4], [10, 8]]
        AB = matrix_matrix_mul(A, B)
        self.assertEqual(AB, [[4.0, 4.0], [10.0, 8.0]])

    def test_matrix_transpose_and_identity(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        AT = matrix_transpose(A)
        self.assertEqual(len(AT), 3)
        self.assertEqual(len(AT[0]), 2)
        self.assertEqual(AT[0], [1.0, 4.0])
        self.assertEqual(AT[1], [2.0, 5.0])
        self.assertEqual(AT[2], [3.0, 6.0])

        I3 = identity_matrix(3)
        self.assertEqual(I3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestQRFactorizationMGS(unittest.TestCase):
    """Tests for Modified Gram-Schmidt QR factorization on synthetic matrices."""

    def setUp(self):
        # Synthetic full-rank 5x4 matrix
        self.A_5x4 = [
            [2.0, 1.0, -3.0, 4.0],
            [1.0, -2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0, -1.0],
            [-1.0, 4.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 5.0],
        ]

    def test_qr_dimensions(self):
        """Verifies that Q is 5x4 and R is 4x4."""
        Q, R, diag_info = qr_mgs(self.A_5x4)
        self.assertEqual(len(Q), 5)
        self.assertEqual(len(Q[0]), 4)
        self.assertEqual(len(R), 4)
        self.assertEqual(len(R[0]), 4)
        self.assertTrue(diag_info["is_full_rank"])

    def test_qr_reconstruction(self):
        """Verifies ||A - QR||_F / ||A||_F is on the order of machine precision."""
        Q, R, _ = qr_mgs(self.A_5x4)
        QR = matrix_matrix_mul(Q, R)
        diff = matrix_sub(self.A_5x4, QR)
        rel_error = frobenius_norm(diff) / frobenius_norm(self.A_5x4)
        self.assertLess(rel_error, 1e-14, f"Reconstruction error too high: {rel_error}")

    def test_q_orthogonality(self):
        """Verifies ||Q^T Q - I_4||_F < 1e-14."""
        Q, _, _ = qr_mgs(self.A_5x4)
        QT = matrix_transpose(Q)
        QT_Q = matrix_matrix_mul(QT, Q)
        I4 = identity_matrix(4)
        diff = matrix_sub(QT_Q, I4)
        ortho_error = frobenius_norm(diff)
        self.assertLess(ortho_error, 1e-14, f"Orthogonality error too high: {ortho_error}")

    def test_r_upper_triangular_structure(self):
        """Verifies that R is strictly upper triangular (R[i][j] == 0 for i > j)."""
        _, R, _ = qr_mgs(self.A_5x4)
        n = len(R)
        for i in range(n):
            for j in range(i):
                self.assertEqual(
                    R[i][j], 0.0,
                    f"R[{i}][{j}] should be 0.0 in upper triangular matrix, got {R[i][j]}"
                )

    def test_r_diagonal_positive(self):
        """Verifies that diagonal entries of R are strictly positive for full-rank matrix."""
        _, R, diag_info = qr_mgs(self.A_5x4)
        self.assertTrue(diag_info["is_full_rank"])
        for i in range(len(R)):
            self.assertGreater(R[i][i], 0.0)


class TestBackSubstitution(unittest.TestCase):
    """Tests for backward substitution on upper-triangular systems."""

    def test_back_substitution_known_system(self):
        """Solves R x = d with known exact solution."""
        R = [
            [3.0, 2.0, -1.0],
            [0.0, 4.0, 1.0],
            [0.0, 0.0, 5.0],
        ]
        x_exact = [2.0, -1.0, 3.0]
        # d = R * x_exact = [3*2 + 2*(-1) - 1*3, 4*(-1) + 1*3, 5*3] = [1, -1, 15]
        d = [1.0, -1.0, 15.0]
        x_computed = back_substitution(R, d)

        for comp, ex in zip(x_computed, x_exact):
            self.assertAlmostEqual(comp, ex, places=14)

    def test_back_substitution_near_singular_raises(self):
        """Raises RankDeficiencyError when pivot is numerically zero."""
        R_singular = [
            [2.0, 1.0],
            [0.0, 1e-18],
        ]
        d = [1.0, 2.0]
        with self.assertRaises(RankDeficiencyError):
            back_substitution(R_singular, d, tol=1e-15)


class TestLeastSquaresSolver(unittest.TestCase):
    """Tests for least-squares resolution via QR."""

    def test_known_least_squares_problem(self):
        """Validates against known overdetermined system with analytical solution:

        A = [[1, 0],
             [0, 1],
             [1, 1]],   b = [1, 1, 0]^T
        Normal equations:
        A^T A = [[2, 1], [1, 2]]
        A^T b = [1, 1]^T
        (A^T A)^{-1} = (1/3) * [[2, -1], [-1, 2]]
        x* = [1/3, 1/3]^T
        """
        A = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        b = [1.0, 1.0, 0.0]

        x_ls, Q, R, diag_info = solve_least_squares_qr(A, b)
        self.assertTrue(diag_info["is_full_rank"])
        self.assertAlmostEqual(x_ls[0], 1.0 / 3.0, places=14)
        self.assertAlmostEqual(x_ls[1], 1.0 / 3.0, places=14)

        # Check least-squares orthogonality indicator: ||A^T (A x* - b)||_2 == 0
        indicators = compute_qr_indicators(A, Q, R, x_ls, b)
        self.assertLess(indicators["eta_LS"], 1e-14)
        # Residual norm: ||[-2/3, -2/3, 2/3]||_2 = sqrt(12/9) = 2/sqrt(3)
        expected_rho = math.sqrt(4.0 / 3.0)
        self.assertAlmostEqual(indicators["rho"], expected_rho, places=14)


class TestRankDeficiencyHandling(unittest.TestCase):
    """Tests proper detection and interruption when rank deficiency occurs."""

    def test_linearly_dependent_columns_detected(self):
        """Matrix where column 3 is an exact linear combination of columns 0, 1, and 2."""
        # Col 0 = [1, 0, 0, 1, 0]
        # Col 1 = [0, 1, 0, 0, 1]
        # Col 2 = [0, 0, 1, 1, 1]
        # Col 3 = Col 0 + Col 1 + Col 2 = [1, 1, 1, 2, 2]
        A_rank_def = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]

        Q, R, diag_info = qr_mgs(A_rank_def)
        self.assertFalse(diag_info["is_full_rank"], "Should detect linear dependency in column 3.")
        self.assertEqual(diag_info["rank_deficiency_col"], 3)

        # Attempting least-squares solve must interrupt and raise RankDeficiencyError
        with self.assertRaises(RankDeficiencyError):
            solve_least_squares_qr(A_rank_def, b)

    def test_column_of_zeros(self):
        """Matrix with a zero column is detected as rank deficient without crash."""
        A_zero_col = [
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 4.0],
            [1.0, 0.0, 1.0],
        ]
        _, _, diag_info = qr_mgs(A_zero_col)
        self.assertFalse(diag_info["is_full_rank"])
        self.assertEqual(diag_info["rank_deficiency_col"], 1)

    def test_invalid_matrix_dimensions(self):
        """Matrix with m < n must raise ValueError."""
        A_under = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2x3
        with self.assertRaises(ValueError):
            qr_mgs(A_under)


class TestMatriculaValidation(unittest.TestCase):
    """Tests synthetic validation and zero replacement rules for student enrollment input."""

    def test_valid_synthetic_matricula(self):
        # 8 non-zero digits
        m = validate_and_process_matricula("87654321")
        self.assertEqual(m, [8, 7, 6, 5, 4, 3, 2, 1])

    def test_zero_substitution_rule(self):
        # Zeros must be replaced by 1s
        m = validate_and_process_matricula("10020034")
        self.assertEqual(m, [1, 1, 1, 2, 1, 1, 3, 4])

    def test_whitespace_handling(self):
        # Leading and trailing whitespace should be stripped
        m = validate_and_process_matricula("  12345678  \n")
        self.assertEqual(m, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_invalid_length_too_short(self):
        with self.assertRaises(ValueError):
            validate_and_process_matricula("1234567")

    def test_invalid_length_too_long(self):
        with self.assertRaises(ValueError):
            validate_and_process_matricula("123456789")

    def test_invalid_characters(self):
        with self.assertRaises(ValueError):
            validate_and_process_matricula("1234567a")


class TestProblem1PipelineSynthetic(unittest.TestCase):
    """Tests the full Problem 1 pipeline with a synthetic 8-digit vector."""

    def test_synthetic_problem1_execution(self):
        # Use synthetic dummy digits: [1, 2, 3, 4, 5, 6, 7, 8]
        m_synthetic = [1, 2, 3, 4, 5, 6, 7, 8]
        A_np, b_np = build_problem1_data(m_synthetic)

        # Verify shapes
        self.assertEqual(A_np.shape, (5, 4))
        self.assertEqual(b_np.shape, (5,))

        # Run pipeline
        res = run_problem1(m_synthetic)
        self.assertTrue(res["diag_info"]["is_full_rank"])
        self.assertTrue(res["is_unique"])

        # Check indicators
        ind = res["indicators"]
        self.assertLess(ind["eta_QR"], 1e-14)
        self.assertLess(ind["eta_Q"], 1e-14)
        self.assertLess(ind["eta_LS"], 1e-12)
        self.assertGreater(ind["rho"], 0.0)


if __name__ == "__main__":
    unittest.main()
