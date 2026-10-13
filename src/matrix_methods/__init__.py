"""Package matrix_methods: Numerical Linear Algebra routines from first principles.

Developed for the Master's program in Applied Mathematics (PPGMAP/UFRGS).
"""

from .linear_algebra import (
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
from .problem1 import (
    build_problem1_data,
    read_and_validate_matricula,
    run_problem1,
    validate_and_process_matricula,
)

__all__ = [
    "RankDeficiencyError",
    "back_substitution",
    "compute_qr_indicators",
    "frobenius_norm",
    "identity_matrix",
    "matrix_matrix_mul",
    "matrix_sub",
    "matrix_transpose",
    "matrix_vector_mul",
    "qr_mgs",
    "solve_least_squares_qr",
    "vector_dot",
    "vector_norm",
    "build_problem1_data",
    "read_and_validate_matricula",
    "run_problem1",
    "validate_and_process_matricula",
]
