"""Automated unit test suite for Problem 2 (Quadratic Least-Squares Fitting).

All tests use purely synthetic, deterministic data and are strictly independent
of any student enrollment number (matrícula).
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

# Ensure src is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from matrix_methods.linear_algebra import (
    compute_qr_indicators,
    qr_mgs,
    solve_least_squares_qr,
)
from matrix_methods.problem2 import (
    build_design_matrix,
    build_problem2_data,
    compute_mse,
    evaluate_polynomial,
    plot_problem2_fit,
    predict_values,
    run_problem2,
)


class TestProblem2DesignMatrix(unittest.TestCase):
    """Verifications for Vandermonde design matrix construction."""

    def test_design_matrix_dimensions(self):
        """Test 1: Verify design matrix X has dimensions m x 3."""
        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        X = build_design_matrix(x_nodes)
        self.assertEqual(len(X), 6)
        for row in X:
            self.assertEqual(len(row), 3)

    def test_design_matrix_columns(self):
        """Test 2: Verify columns correspond precisely to [x_i^2, x_i, 1.0]."""
        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        X = build_design_matrix(x_nodes)
        expected_rows = [
            [1.0, 1.0, 1.0],
            [4.0, 2.0, 1.0],
            [9.0, 3.0, 1.0],
            [16.0, 4.0, 1.0],
            [25.0, 5.0, 1.0],
            [36.0, 6.0, 1.0],
        ]
        for i, (actual, expected) in enumerate(zip(X, expected_rows)):
            for j in range(3):
                self.assertAlmostEqual(
                    actual[j],
                    expected[j],
                    places=12,
                    msg=f"Mismatch at row {i}, col {j}: {actual[j]} != {expected[j]}",
                )


class TestQuadraticFittingSynthetic(unittest.TestCase):
    """Tests for polynomial fitting and coefficient recovery on synthetic data."""

    def test_recovery_of_exact_quadratic_polynomial(self):
        """Test 3: Perfect recovery of known coefficients for exact quadratic data."""
        # Target polynomial: p(x) = 2.5 * x^2 - 4.2 * x + 7.8
        true_a, true_b, true_c = 2.5, -4.2, 7.8
        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_exact = [evaluate_polynomial(true_a, true_b, true_c, xi) for xi in x_nodes]

        X = build_design_matrix(x_nodes)
        theta_computed, Q, R, diag_info = solve_least_squares_qr(X, y_exact)

        comp_a, comp_b, comp_c = theta_computed[0], theta_computed[1], theta_computed[2]

        self.assertAlmostEqual(comp_a, true_a, places=10)
        self.assertAlmostEqual(comp_b, true_b, places=10)
        self.assertAlmostEqual(comp_c, true_c, places=10)

        # In an exact quadratic model, residuals and MSE should be near machine precision
        y_pred = predict_values(theta_computed, x_nodes)
        mse = compute_mse(y_exact, y_pred)
        self.assertLess(mse, 1e-20)

    def test_non_exact_quadratic_positive_error(self):
        """Test 4: Fitting noisy/non-quadratic data must result in positive MSE."""
        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Data with artificial non-quadratic perturbation
        y_perturbed = [1.0, 4.0, 2.0, 8.0, 3.0, 7.0]

        X = build_design_matrix(x_nodes)
        theta_computed, Q, R, diag_info = solve_least_squares_qr(X, y_perturbed)

        y_pred = predict_values(theta_computed, x_nodes)
        mse = compute_mse(y_perturbed, y_pred)

        self.assertGreater(mse, 0.0)
        # Check that predicted values differ from true values
        diffs = [abs(yt - yp) for yt, yp in zip(y_perturbed, y_pred)]
        self.assertGreater(max(diffs), 0.1)

    def test_manual_mse_calculation(self):
        """Test 5: Verify explicit manual MSE calculation against expected analytical value."""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.5, 2.5, 2.0, 5.0]
        # Differences: [-0.5, -0.5, 1.0, -1.0]
        # Squared differences: [0.25, 0.25, 1.0, 1.0]
        # Sum = 2.5, Mean = 2.5 / 4 = 0.625
        expected_mse = 0.625
        calculated_mse = compute_mse(y_true, y_pred)
        self.assertAlmostEqual(calculated_mse, expected_mse, places=14)

    def test_qr_indicators_on_quadratic_system(self):
        """Tests 6 & 7: Verify QR factorization metrics and LS optimality condition."""
        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_noisy = [2.0, 1.0, 4.0, 3.0, 5.0, 2.0]

        X = build_design_matrix(x_nodes)
        theta_ls, Q, R, diag_info = solve_least_squares_qr(X, y_noisy)

        indicators = compute_qr_indicators(X, Q, R, theta_ls, y_noisy)

        # 1. Relative QR reconstruction error: ||X - QR||_F / ||X||_F < 1e-14
        self.assertLess(indicators["eta_QR"], 1e-14)

        # 2. Orthogonality error: ||Q^T Q - I_3||_F < 1e-14
        self.assertLess(indicators["eta_Q"], 1e-14)

        # 3. Least-squares first-order optimality condition: ||X^T (X theta - y)||_2 < 1e-12
        self.assertLess(indicators["eta_LS"], 1e-12)

        # 4. Consistency: MSE * 6 == rho^2
        rho = indicators["rho"]
        y_pred = predict_values(theta_ls, x_nodes)
        mse = compute_mse(y_noisy, y_pred)
        self.assertAlmostEqual(mse * len(y_noisy), rho * rho, places=12)

    def test_independence_from_numpy_linalg(self):
        """Test 8: Verify that solve_least_squares_qr does NOT invoke numpy.linalg."""
        import numpy.linalg as npla

        # Temporarily monkey-patch numpy.linalg methods to raise if invoked
        def forbidden_call(*args, **kwargs):
            raise AssertionError("Chamada proibida a rotina de numpy.linalg!")

        orig_qr = npla.qr
        orig_lstsq = npla.lstsq
        orig_solve = npla.solve

        try:
            npla.qr = forbidden_call
            npla.lstsq = forbidden_call
            npla.solve = forbidden_call

            x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            y_synthetic = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            X = build_design_matrix(x_nodes)

            # Must run cleanly without raising AssertionError
            theta_computed, Q, R, diag_info = solve_least_squares_qr(X, y_synthetic)
            self.assertEqual(len(theta_computed), 3)
            self.assertTrue(diag_info["is_full_rank"])
        finally:
            npla.qr = orig_qr
            npla.lstsq = orig_lstsq
            npla.solve = orig_solve


class TestProblem2EndToEnd(unittest.TestCase):
    """End-to-end tests for Problem 2 using synthetic matricula data."""

    def test_run_problem2_synthetic(self):
        # Synthetic processed matricula: 8 digits
        synthetic_m = [2, 4, 1, 3, 5, 2, 7, 6]
        results = run_problem2(synthetic_m)

        self.assertEqual(len(results["x"]), 6)
        self.assertEqual(len(results["y"]), 6)
        self.assertEqual(len(results["theta"]), 3)
        self.assertTrue(results["is_unique"])
        self.assertGreater(len(results["uniqueness_justification"]), 50)
        self.assertGreaterEqual(results["mse"], 0.0)

        # Check diagonal entries of R are strictly positive and >> tol
        diag_R = results["diag_info"]["diagonal_entries"]
        tol = results["diag_info"]["tolerance"]
        self.assertEqual(len(diag_R), 3)
        for val in diag_R:
            self.assertGreater(val, 1e-1)
            self.assertGreater(val, tol * 1e10)

    def test_plot_problem2_fit_export(self):
        """Verifies that plot_problem2_fit exports valid PDF and PNG files."""
        x_pts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_pts = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
        theta = [0.1, 0.5, 0.8]
        mse = 0.45

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test_fit.pdf")
            png_path = os.path.join(tmpdir, "test_fit.png")

            paths = plot_problem2_fit(x_pts, y_pts, theta, mse, output_paths=[pdf_path, png_path])

            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 1000)


if __name__ == "__main__":
    unittest.main()
