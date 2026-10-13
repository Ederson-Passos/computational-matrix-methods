"""Problem 1 execution logic and data generation.

Implements the data construction and problem resolution for Problem 1
of Computational Work 3 (TC3), preserving the exact formulas and array
construction requested by the professor in TC3.pdf.
"""

from __future__ import annotations

import getpass
from typing import Any
import numpy as np

from .linear_algebra import (
    compute_qr_indicators,
    qr_mgs,
    solve_least_squares_qr,
    to_float_matrix,
    to_float_vector,
)


def validate_and_process_matricula(matricula_str: str) -> list[int]:
    """Validates and processes the 8-digit student enrollment number string.

    Rules:
        - Must contain exactly 8 decimal digits.
        - Each digit zero is replaced by 1, according to the assignment instructions:
          m = [1 if x == 0 else x for x in m].
        - Does NOT leak or print the input digits.

    Args:
        matricula_str: The raw student enrollment string.

    Returns:
        A list of 8 integers [m1, m2, ..., m8] with zeros replaced by ones.

    Raises:
        ValueError: If input length is not 8 or contains non-decimal characters.
    """
    cleaned = matricula_str.strip()
    if len(cleaned) != 8:
        raise ValueError(f"Entrada inválida: esperado 8 dígitos decimais, recebido comprimento {len(cleaned)}.")
    if not cleaned.isdigit():
        raise ValueError("Entrada inválida: a matrícula deve conter exclusivamente dígitos decimais (0-9).")

    # Step from assignment:
    # m = [int(d) for d in matricula]
    # m = [1 if x == 0 else x for x in m]
    m_raw = [int(d) for d in cleaned]
    m_processed = [1 if x == 0 else x for x in m_raw]
    return m_processed


def read_and_validate_matricula(prompt_msg: str = "Digite a matrícula (8 dígitos numéricos): ") -> list[int]:
    """Interactively reads the student enrollment number from terminal using getpass.

    getpass prevents echoing the digits on screen, ensuring data privacy.
    If stdin is redirected/piped (non-interactive), reads securely from sys.stdin.

    Returns:
        The processed list of 8 integers [m1, m2, ..., m8].
    """
    import sys
    if sys.stdin.isatty():
        raw_input_str = getpass.getpass(prompt_msg).strip()
    else:
        # Piped/redirected stream fallback
        raw_input_str = sys.stdin.readline().strip()
    return validate_and_process_matricula(raw_input_str)


def build_problem1_data(m_processed: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Constructs matrix A and vector b as defined in Problem 1 of TC3.pdf.

    Preserves exact formula and code structure from assignment:
        m1, m2, m3, m4, m5, m6, m7, m8 = m
        A = np.array([
            [m1+2,  m2,   -m3,   m4 ],
            [m2+1, -m3,    m4+2, m5 ],
            [m3+2, -m4,    m5+1, -m6],
            [-m4,   m5+2,  m6,   m7 ],
            [-m5,   m6+1,  m7+2, m8 ]
        ], dtype=float)

        b = np.array([
            m1+m2,
            -m3,
            m4+m5,
            -m6,
            m7+m8
        ], dtype=float)

    Args:
        m_processed: List of 8 integer digits [m1, ..., m8].

    Returns:
        A tuple of (A, b) as numpy.ndarray instances.
    """
    m1, m2, m3, m4, m5, m6, m7, m8 = m_processed

    # =====================================================
    # MATRIZ A (Preservado conforme TC3.pdf)
    # =====================================================
    A = np.array(
        [
            [m1 + 2, m2, -m3, m4],
            [m2 + 1, -m3, m4 + 2, m5],
            [m3 + 2, -m4, m5 + 1, -m6],
            [-m4, m5 + 2, m6, m7],
            [-m5, m6 + 1, m7 + 2, m8],
        ],
        dtype=float,
    )

    # =====================================================
    # VETOR b (Preservado conforme TC3.pdf)
    # =====================================================
    b = np.array(
        [
            m1 + m2,
            -m3,
            m4 + m5,
            -m6,
            m7 + m8,
        ],
        dtype=float,
    )

    return A, b


def run_problem1(m_processed: list[int], tol: float | None = None) -> dict[str, Any]:
    """Executes the full pipeline for Problem 1 using Modified Gram-Schmidt QR.

    Steps:
        1. Construct A and b via build_problem1_data.
        2. Convert to pure Python float structures.
        3. Perform reduced QR factorization via qr_mgs.
        4. Solve the least-squares system R x = Q^T b.
        5. Evaluate the 4 verification indicators.
        6. Return complete structured results.

    Args:
        m_processed: 8-digit processed list.
        tol: Optional tolerance for rank-deficiency detection.

    Returns:
        A dictionary containing all inputs, factors, solutions, indicators,
        and rank diagnostic assessments.
    """
    A_np, b_np = build_problem1_data(m_processed)
    A_list = to_float_matrix(A_np)
    b_list = to_float_vector(b_np)

    # Solve least squares via MGS QR
    x_ls, Q, R, diag_info = solve_least_squares_qr(A_list, b_list, tol=tol)

    # Evaluate verification indicators
    indicators = compute_qr_indicators(A_list, Q, R, x_ls, b_list)

    # Uniqueness assessment:
    # rank(A) = 4 iff all 4 diagonal entries of R are numerically non-zero
    is_unique = diag_info["is_full_rank"]
    uniqueness_justification = (
        "Como a matriz A (5x4) possui posto completo por colunas (rank(A) = 4), "
        "o núcleo de A é trivial (ker(A) = {0}), a matriz R é não-singular (invertível) "
        "e a matriz de Gram A^T A é estritamente definida positiva. Logo, a solução "
        "de mínimos quadrados x* = R^{-1} Q^T b é ÚNICA."
        if is_unique
        else "Posto deficiente detectado: existem infinitas soluções de mínimos quadrados."
    )

    return {
        "A": A_list,
        "b": b_list,
        "Q": Q,
        "R": R,
        "x_ls": x_ls,
        "diag_info": diag_info,
        "indicators": indicators,
        "is_unique": is_unique,
        "uniqueness_justification": uniqueness_justification,
    }


def format_matrix(M: list[list[float]], precision: int = 8) -> str:
    """Formats a 2D matrix into an aligned string representation."""
    lines = []
    for row in M:
        formatted_row = "  ".join(f"{val:+.{precision}e}" for val in row)
        lines.append(f"  [ {formatted_row} ]")
    return "\n".join(lines)


def format_vector(v: list[float], precision: int = 8) -> str:
    """Formats a 1D vector into an aligned string representation."""
    formatted = "  ".join(f"{val:+.{precision}e}" for val in v)
    return f"[ {formatted} ]"


def format_results_report(res: dict[str, Any]) -> str:
    """Generates the comprehensive terminal output report required by Section 9."""
    ind = res["indicators"]
    diag = res["diag_info"]

    report_lines = [
        "=" * 78,
        "TRABALHO COMPUTACIONAL 3 — PROBLEMA 1: FATORAÇÃO QR E MÍNIMOS QUADRADOS",
        "=" * 78,
        "",
        "1. MATRIZ DE COEFICIENTES A (5x4) E VETOR DE TERMOS INDEPENDENTES b (5x1):",
        "-" * 78,
        "Matriz A:",
        format_matrix(res["A"]),
        "",
        "Vetor b:",
        format_vector(res["b"]),
        "",
        "2. FATORAÇÃO QR REDUZIDA VIA GRAM-SCHMIDT MODIFICADO (MGS):",
        "-" * 78,
        "Matriz Q (5x4) [Colunas Ortonormais]:",
        format_matrix(res["Q"]),
        "",
        "Matriz R (4x4) [Triangular Superior]:",
        format_matrix(res["R"]),
        "",
        "3. SOLUÇÃO DE MÍNIMOS QUADRADOS x* (4x1) [R x* = Q^T b]:",
        "-" * 78,
        format_vector(res["x_ls"]),
        "",
        "4. VETOR RESIDUAL r = A x* - b (5x1):",
        "-" * 78,
        format_vector(ind["residual_vector"]),
        "",
        "5. INDICADORES NUMÉRICOS E CRITÉRIOS DE CONVERGÊNCIA:",
        "-" * 78,
        f"  * Norma Euclidiana do Resíduo (rho = ||A x* - b||_2):",
        f"    {ind['rho']:.10e}",
        f"    Significado: Distância geométrica entre b e o espaço coluna Col(A). Em sistemas",
        f"    sobredeterminados incompatíveis, rho > 0 é esperado e reflete a projeção ortogonal.",
        "",
        f"  * Erro Relativo de Reconstrução QR (eta_QR = ||A - QR||_F / ||A||_F):",
        f"    {ind['eta_QR']:.10e}",
        f"    Significado: Mede a precisão da fatoração A ~ QR. Valores próximos à precisão da",
        f"    máquina (~10^-16) confirmam a exatidão algébrica da decomposição.",
        "",
        f"  * Erro de Ortogonalidade de Q (eta_Q = ||Q^T Q - I_4||_F):",
        f"    {ind['eta_Q']:.10e}",
        f"    Significado: Mede o desvio da ortonormalidade das colunas de Q.",
        "",
        f"  * Verificação da Condição Normal de Mínimos Quadrados (eta_LS = ||A^T(A x* - b)||_2):",
        f"    {ind['eta_LS']:.10e}",
        f"    Significado: Condição de otimalidade de 1ª ordem (gradiente nulo). Confirma que o",
        f"    resíduo é estritamente ortogonal a Col(A), garantindo que x* minimiza ||A x - b||_2.",
        "",
        "6. DIAGNÓSTICO NUMÉRICO DE POSTO E UNICIDADE DA SOLUÇÃO:",
        "-" * 78,
        f"  * Posto completo por colunas (rank = 4): {'SIM' if diag['is_full_rank'] else 'NÃO'}",
        f"  * Tolerância numérica de posto: {diag['tolerance']:.10e}",
        f"  * Entradas diagonais de R: {[f'{d:.6e}' for d in diag['diagonal_entries']]}",
        f"  * Diagnóstico de Unicidade:",
        f"    {res['uniqueness_justification']}",
        "",
        "NOTA METODOLÓGICA:",
        "O algoritmo de Gram-Schmidt Modificado sem pivotamento foi utilizado conforme a",
        "metodologia do trabalho. Ressalta-se que, para matrizes arbitrárias severamente mal",
        "condicionadas, o MGS sem pivotamento pode sofrer perda de ortogonalidade e não constitui",
        "um teste de posto revelador garantido (Rank-Revealing QR). Para robustez numérica universal,",
        "recomenda-se QR com pivotamento de colunas (Businger-Golub) ou Decomposição em Valores Singulares (SVD).",
        "=" * 78,
    ]
    return "\n".join(report_lines)
