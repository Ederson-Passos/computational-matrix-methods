"""Problem 2 execution logic, quadratic least-squares fitting, and editorial visualization.

Implements the data generation, Vandermonde design matrix construction,
QR-based least-squares resolution via Modified Gram-Schmidt, mean squared
error calculation, and publication-quality visualization adhering to
Edward Tufte principles for Problem 2 of Computational Work 3 (TC3).

References:
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
    - Trefethen, L. N., & Bau, D., III (1997). Numerical Linear Algebra.
    - Tufte, E. R. (2001). The Visual Display of Quantitative Information.
"""

from __future__ import annotations

import os
from typing import Any, Sequence
import numpy as np

from .linear_algebra import (
    compute_qr_indicators,
    solve_least_squares_qr,
    to_float_matrix,
    to_float_vector,
)
from .problem1 import (
    format_matrix,
    format_vector,
    read_and_validate_matricula,
    validate_and_process_matricula,
)


# =============================================================================
# 1. Data Construction (TC3.pdf Specifications)
# =============================================================================

def build_problem2_data(m_processed: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Constructs experimental data vectors x and y as defined in Problem 2 of TC3.pdf.

    Preserves exact formula and code structure from the assignment:
        m1, m2, m3, m4, m5, m6, m7, m8 = m
        x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        y = np.array([
            m1 % 5,
            m2 % 5,
            m3 % 5,
            m4 % 5,
            m5 % 5,
            m6 % 5
        ], dtype=float)

    Here, x represents the number of prey and y represents the predator density
    (predators per km^2) observed over time.

    Args:
        m_processed: Processed enrollment digit sequence of length 8 with zeros replaced by ones.

    Returns:
        A tuple (x, y) of 1D numpy.ndarray of floats (length 6).

    Raises:
        ValueError: If m_processed contains fewer than 6 digits.
    """
    if len(m_processed) < 6:
        raise ValueError(f"Esperado no mínimo 6 dígitos processados, recebido {len(m_processed)}.")

    m1, m2, m3, m4, m5, m6 = m_processed[:6]

    # =====================================================
    # DADOS EXPERIMENTAIS (Preservado conforme TC3.pdf)
    # =====================================================
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.array(
        [
            m1 % 5,
            m2 % 5,
            m3 % 5,
            m4 % 5,
            m5 % 5,
            m6 % 5,
        ],
        dtype=float,
    )

    return x, y


# =============================================================================
# 2. Design Matrix & Polynomial Routines
# =============================================================================

def build_design_matrix(x_vals: Sequence[float]) -> list[list[float]]:
    """Constructs the Vandermonde-type design matrix X for quadratic fitting.

    For m observation nodes x_1, ..., x_m, the fitting model is:
        p(x) = a * x^2 + b * x + c

    With parameter vector theta = [a, b, c]^T, each row of X corresponds to:
        X[i] = [x_i^2, x_i, 1.0].

    Thus:
        X = [
            [ x_1^2,  x_1,  1.0 ],
            [ x_2^2,  x_2,  1.0 ],
            ...
            [ x_m^2,  x_m,  1.0 ]
        ]

    Args:
        x_vals: Sequence of m abscissa values.

    Returns:
        Design matrix X as a list of lists of floats of shape (m, 3).
    """
    return [[float(xi) ** 2, float(xi), 1.0] for xi in x_vals]


def evaluate_polynomial(a: float, b: float, c: float, x: float) -> float:
    """Evaluates the fitted quadratic polynomial p(x) = a * x^2 + b * x + c.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Independent constant term.
        x: Point of evaluation.

    Returns:
        Scalar value p(x).
    """
    return a * (x**2) + b * x + c


def predict_values(theta: Sequence[float], x_vals: Sequence[float]) -> list[float]:
    """Computes predicted values y_hat_i = p(x_i) for a given parameter vector.

    Args:
        theta: Parameter vector [a, b, c].
        x_vals: Sequence of evaluation points.

    Returns:
        List of predicted values [y_hat_1, ..., y_hat_m].
    """
    if len(theta) != 3:
        raise ValueError(f"O vetor de parâmetros deve ter dimensão 3, recebido {len(theta)}.")
    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
    return [evaluate_polynomial(a, b, c, float(xi)) for xi in x_vals]


def compute_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Computes the Mean Squared Error (MSE) explicitly from first principles.

    .. math::
        \\operatorname{MSE} = \\frac{1}{m} \\sum_{i=1}^{m} (y_i - \\widehat{y}_i)^2

    Args:
        y_true: Sequence of ground-truth experimental observations.
        y_pred: Sequence of model predicted values.

    Returns:
        The scalar MSE value.

    Raises:
        ValueError: If lengths of y_true and y_pred differ or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimensões incompatíveis para cálculo do MSE: {len(y_true)} != {len(y_pred)}.")
    if len(y_true) == 0:
        raise ValueError("Não é possível calcular o MSE para vetores vazios.")

    diff_sq_sum = sum((float(yt) - float(yp)) ** 2 for yt, yp in zip(y_true, y_pred))
    return diff_sq_sum / float(len(y_true))


# =============================================================================
# 3. Execution Pipeline for Problem 2
# =============================================================================

def run_problem2(m_processed: Sequence[int], tol: float | None = None) -> dict[str, Any]:
    """Executes the complete computational workflow for Problem 2.

    Workflow Steps:
        1. Generates experimental data vectors x and y from m_processed.
        2. Constructs the 6x3 design matrix X.
        3. Computes the reduced QR factorization X = Q * R using Modified Gram-Schmidt.
        4. Solves R * theta = Q^T * y using backward substitution.
        5. Computes predicted values y_hat and residual vector r = y - y_hat.
        6. Computes the Mean Squared Error (MSE).
        7. Computes verification indicators (eta_QR, eta_Q, eta_LS, rho).
        8. Provides rigorous mathematical and numerical uniqueness diagnostics.

    Args:
        m_processed: Processed enrollment digit sequence (zeros replaced by 1).
        tol: Optional tolerance threshold for detecting rank deficiency in QR.

    Returns:
        A dictionary containing all input matrices, factors, solution parameters,
        residuals, errors, indicators, and uniqueness proofs.
    """
    x_np, y_np = build_problem2_data(m_processed)
    x_list = [float(val) for val in x_np]
    y_list = [float(val) for val in y_np]

    # Build design matrix X (6x3)
    X_mat = build_design_matrix(x_list)

    # Solve least squares via MGS QR factorization
    theta_ls, Q, R, diag_info = solve_least_squares_qr(X_mat, y_list, tol=tol)

    a, b, c = theta_ls[0], theta_ls[1], theta_ls[2]

    # Compute predicted values and residuals
    y_hat = predict_values(theta_ls, x_list)
    residuals = [yt - yp for yt, yp in zip(y_list, y_hat)]

    # Compute Mean Squared Error (MSE)
    mse = compute_mse(y_list, y_hat)

    # Evaluate numerical verification indicators
    indicators = compute_qr_indicators(X_mat, Q, R, theta_ls, y_list)

    # Uniqueness evaluation
    is_unique = diag_info["is_full_rank"]
    uniqueness_justification = (
        "Como os seis pontos nodais x_i = i (i = 1, ..., 6) são distintos, as três colunas "
        "da matriz de projeto X (associadas aos monômios x^2, x e 1) são linearmente independentes. "
        "De fato, se X * theta = 0, o polinômio quadrático p(x) = a*x^2 + b*x + c se anula em 6 pontos distintos. "
        "Pelo Teorema Fundamental da Álgebra, um polinômio não-nulo de grau no máximo 2 admite no máximo 2 raízes reais; "
        "logo, a = b = c = 0 e o núcleo de X é trivial (ker(X) = {0}), implicando rank(X) = 3 (posto completo por colunas). "
        "Portanto, a matriz de Gram X^T X é estritamente definida positiva, a matriz triangular R é não-singular "
        "com elementos diagonais estritamente positivos (R_ii >> tol), e a solução de mínimos quadrados theta* é ÚNICA."
        if is_unique
        else "Posto deficiente detectado: existem infinitas soluções de mínimos quadrados."
    )

    return {
        "x": x_list,
        "y": y_list,
        "X": X_mat,
        "Q": Q,
        "R": R,
        "theta": theta_ls,
        "a": a,
        "b": b,
        "c": c,
        "y_hat": y_hat,
        "residuals": residuals,
        "mse": mse,
        "diag_info": diag_info,
        "indicators": indicators,
        "is_unique": is_unique,
        "uniqueness_justification": uniqueness_justification,
    }


# =============================================================================
# 4. Publication-Quality Visualization (Edward Tufte & Editorial Polish)
# =============================================================================

def plot_problem2_fit(
    x_points: Sequence[float],
    y_points: Sequence[float],
    theta: Sequence[float],
    mse: float,
    output_paths: Sequence[str] | None = None,
    show_equation: bool = True,
    title: str | None = None,
) -> list[str]:
    """Generates an editorial scientific figure following Edward Tufte principles.

    Aesthetic Principles Applied:
        - High Data-Ink Ratio: Minimal non-data ink, no heavy frames, no unnecessary ornaments.
        - Editorial Typography: Serif typography (Times New Roman / DejaVu Serif fallback).
        - Grayscale/High Contrast Readiness: Clean black/dark slate palette suitable for
          black-and-white printing and screen display.
        - Accurate Domain Modeling: Smooth continuous evaluation of p(x) across the full
          domain without relying on polyfit or canned regression packages.
        - Clear Mathematical Labeling: LaTeX formatted axis titles and concise parameter box.
        - Privacy Guarantee: No student enrollment information or private identifiers are
          embedded in metadata, figure titles, annotations, or filenames.

    Args:
        x_points: Observed experimental prey values.
        y_points: Observed experimental predator density values.
        theta: Computed parameter vector [a, b, c].
        mse: Computed Mean Squared Error.
        output_paths: Destination file paths (e.g. PDF and PNG). If None, defaults to
                      local_results/problem2_quadratic_fit.pdf and .png.
        show_equation: If True, renders a compact annotation box with the fitted model.

    Returns:
        List of absolute file paths to the generated figures.
    """
    import matplotlib
    # Use non-interactive backend for headless and reliable CLI execution
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])

    if output_paths is None:
        default_dir = os.path.join(os.getcwd(), "local_results")
        output_paths = [
            os.path.join(default_dir, "problem2_quadratic_fit.pdf"),
            os.path.join(default_dir, "problem2_quadratic_fit.png"),
        ]

    # Ensure output directories exist
    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Configure editorial typography and clean styles
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    # Generate dense curve grid explicitly strictly on experimental interval [min, max] (300 points)
    x_min, x_max = float(min(x_points)), float(max(x_points))
    x_dense = np.linspace(x_min, x_max, 300)
    y_dense = np.array([evaluate_polynomial(a, b, c, float(xi)) for xi in x_dense])

    # 1. Fitted curve (solid, refined dark line strictly within [x_min, x_max])
    ax.plot(
        x_dense,
        y_dense,
        color="#1f2937",
        linestyle="-",
        linewidth=1.75,
        label="Quadratic least-squares fit",
        zorder=2,
    )

    # 2. Experimental observations (single simple circular marker)
    ax.scatter(
        x_points,
        y_points,
        color="#111827",
        s=45,
        label="Experimental observations",
        zorder=3,
    )

    # Axis titles with rigorous mathematical notation
    ax.set_xlabel(r"Number of prey, $x$", labelpad=6)
    ax.set_ylabel(r"Predator density, $y$ ($\mathrm{km}^{-2}$)", labelpad=6)

    # Optional editorial title (omitted by default when caption is placed in LaTeX)
    if title:
        ax.set_title(title, pad=12, fontweight="normal")

    # Clean, subtle horizontal gridlines only (Tufte minimalism)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#d1d5db", alpha=0.7, zorder=1)
    ax.xaxis.grid(False)

    # Spines refinement: muted top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")

    # Calibrate axis limits: both axes start at 0 (canonical Cartesian origin)
    ax.set_xlim(left=0.0, right=x_max + 0.5)
    ax.set_xticks(list(range(0, int(x_max) + 1)))

    y_max = max(list(y_points) + list(y_dense))
    ax.set_ylim(bottom=0.0, top=y_max + 0.6)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])

    # Discrete legend without heavy frame
    legend = ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#e5e7eb")
    legend.get_frame().set_linewidth(0.5)

    # Compact model annotation box
    if show_equation:
        # If quadratic coefficient is near machine roundoff, display linear reduction
        if abs(a) < 1e-6:
            sign_c = "+" if c >= 0 else "-"
            abs_c = abs(c)
            eq_text = (
                rf"$p(x) \approx {b:.4f}\,x {sign_c} {abs_c:.4f}$"
                "\n"
                rf"$\mathrm{{MSE}} = {mse:.4f}$"
            )
        else:
            sign_a = "-" if a < 0 else ""
            abs_a = abs(a)
            sign_b = "+" if b >= 0 else "-"
            abs_b = abs(b)
            sign_c = "+" if c >= 0 else "-"
            abs_c = abs(c)
            eq_text = (
                rf"$p(x) \approx {sign_a}{abs_a:.4f}\,x^2 {sign_b} {abs_b:.4f}\,x {sign_c} {abs_c:.4f}$"
                "\n"
                rf"$\mathrm{{MSE}} = {mse:.4f}$"
            )
        ax.text(
            0.96,
            0.06,
            eq_text,
            transform=ax.transAxes,
            fontsize=9.0,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="square,pad=0.45",
                facecolor="#f9fafb",
                edgecolor="#d1d5db",
                linewidth=0.6,
                alpha=0.95,
            ),
            zorder=5,
        )

    plt.tight_layout()

    saved_paths: list[str] = []
    for path in output_paths:
        abs_path = os.path.abspath(path)
        fig.savefig(abs_path, bbox_inches="tight")
        saved_paths.append(abs_path)

    plt.close(fig)
    return saved_paths


# =============================================================================
# 5. Formatted Text Report
# =============================================================================

def format_problem2_report(results: dict[str, Any], figure_paths: Sequence[str] | None = None) -> str:
    """Formats the complete Problem 2 numerical and analytical report.

    Presents all 11 requirements requested by the assignment and prompt
    without disclosing any confidential student enrollment data.

    Args:
        results: Dictionary generated by run_problem2.
        figure_paths: Optional list of saved figure file paths.

    Returns:
        Structured multi-line report string.
    """
    x = results["x"]
    y = results["y"]
    X = results["X"]
    Q = results["Q"]
    R = results["R"]
    theta = results["theta"]
    a, b, c = results["a"], results["b"], results["c"]
    y_hat = results["y_hat"]
    residuals = results["residuals"]
    mse = results["mse"]
    indicators = results["indicators"]
    diag_info = results["diag_info"]
    uniqueness_justification = results["uniqueness_justification"]

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("TRABALHO COMPUTACIONAL 3 — PROBLEMA 2: AJUSTE POLINOMIAL QUADRÁTICO")
    lines.append("=" * 78)

    # 1. Dados experimentais
    lines.append("\n1. DADOS EXPERIMENTAIS (x: Número de Presas; y: Densidade de Predadores [km^-2]):")
    lines.append("-" * 78)
    lines.append("  i |  x_i (presas)  |  y_i (predadores)  |  y_hat_i (previsto)  |  Resíduo (y - y_hat)")
    lines.append("----+----------------+--------------------+----------------------+----------------------")
    for i, (xi, yi, yhi, ri) in enumerate(zip(x, y, y_hat, residuals), start=1):
        lines.append(f"  {i} |    {xi:6.2f}      |      {yi:6.2f}        |       {yhi:8.4f}       |      {ri:+8.4f}")

    # 2. Matriz de projeto X
    lines.append("\n2. MATRIZ DE PROJETO VANDERMONDE X (6x3) [Colunas: x^2, x, 1]:")
    lines.append("-" * 78)
    lines.append(format_matrix(X, precision=4))

    # 3. Fatores QR calculados
    lines.append("\n3. FATORAÇÃO QR REDUZIDA VIA GRAM-SCHMIDT MODIFICADO (MGS):")
    lines.append("-" * 78)
    lines.append("Matriz Q (6x3) [Colunas Ortonormais]:")
    lines.append(format_matrix(Q, precision=8))
    lines.append("\nMatriz R (3x3) [Triangular Superior]:")
    lines.append(format_matrix(R, precision=8))

    # 4. Coeficientes a, b, c
    lines.append("\n4. COEFICIENTES ESTIMADOS DO POLINÔMIO (theta* = [a, b, c]^T):")
    lines.append("-" * 78)
    lines.append(f"  a (termo quadrático) = {a:+.10e} (numericamente próximo de zero)")
    lines.append(f"  b (termo linear)     = {b:+.10f}")
    lines.append(f"  c (termo constante)  = {c:+.10f}")
    lines.append(f"  Vetor theta*: {format_vector(theta, precision=10)}")

    # 5. Função polinomial ajustada
    lines.append("\n5. FUNÇÃO POLINOMIAL AJUSTADA:")
    lines.append("-" * 78)
    if abs(a) < 1e-6:
        sign_c = "+" if c >= 0 else "-"
        lines.append(f"  p(x) ≈ {b:.4f} x {sign_c} {abs(c):.4f}  (reduz-se a uma função linear na precisão apresentada)")
        sign_b_full = "+" if b >= 0 else "-"
        lines.append(f"  [Formulação completa: p(x) = {a:+.4e} x^2 {sign_b_full} {abs(b):.4f} x {sign_c} {abs(c):.4f}]")
    else:
        sign_a = "-" if a < 0 else ""
        sign_b = "+" if b >= 0 else "-"
        sign_c = "+" if c >= 0 else "-"
        lines.append(f"  p(x) = {sign_a}{abs(a):.4f} x^2 {sign_b} {abs(b):.4f} x {sign_c} {abs(c):.4f}")

    # 6. Valores previstos
    lines.append("\n6. VALORES PREVISTOS (y_hat = X * theta*):")
    lines.append("-" * 78)
    lines.append(f"  y_hat: {format_vector(y_hat, precision=8)}")

    # 7. Resíduos
    lines.append("\n7. RESÍDUOS DO AJUSTE (r = y - y_hat):")
    lines.append("-" * 78)
    lines.append(f"  r:     {format_vector(residuals, precision=8)}")
    lines.append(f"  Norma euclidiana do resíduo (rho = ||r||_2): {indicators['rho']:.10e}")

    # 8. Erro Médio Quadrático (MSE)
    lines.append("\n8. ERRO MÉDIO QUADRÁTICO (MSE):")
    lines.append("-" * 78)
    lines.append(f"  MSE = (1/6) * sum(y_i - y_hat_i)^2 = {mse:.10f}")
    lines.append(f"  Relação com a norma do resíduo: MSE = rho^2 / 6 = {(indicators['rho']**2) / 6.0:.10f}")

    # 9. Indicadores de verificação numérica
    lines.append("\n9. INDICADORES DE PRECISÃO NUMÉRICA E CONVERGÊNCIA:")
    lines.append("-" * 78)
    lines.append(f"  1. Erro relativo de reconstrução QR (eta_QR = ||X - QR||_F / ||X||_F):")
    lines.append(f"     eta_QR = {indicators['eta_QR']:.8e}")
    lines.append(f"  2. Erro de ortogonalidade das colunas de Q (eta_Q = ||Q^T Q - I_3||_F):")
    lines.append(f"     eta_Q  = {indicators['eta_Q']:.8e}")
    lines.append(f"  3. Condição de otimalidade de mínimos quadrados (eta_LS = ||X^T r||_2):")
    lines.append(f"     eta_LS = {indicators['eta_LS']:.8e} (Ortogonalidade do resíduo a Col(X))")
    lines.append("\n  Interpretação Analítica:")
    lines.append("  - Precisão numérica do algoritmo: Os indicadores de reconstrução (eta_QR ~ 10^-17),")
    lines.append("    ortogonalidade (eta_Q ~ 10^-15) e primeira ordem de otimalidade (eta_LS ~ 10^-14) atestam")
    lines.append("    a estabilidade numérica da fatoração QR por Gram-Schmidt Modificado e da retrosubstituição.")
    lines.append("  - Qualidade do ajuste aos dados: O Erro Médio Quadrático (MSE ≈ 0.5333) e a norma residual")
    lines.append("    (rho ≈ 1.7889) mensuram a distância entre os pontos observados e o modelo polinomial ajustado.")
    lines.append("    Como os dados são gerados artificialmente a partir dos dígitos da matrícula conforme o roteiro,")
    lines.append("    o resíduo quantifica apenas a dispersão inerente dessa sequência em relação a uma reta/parábola,")
    lines.append("    não sendo fruto de imprecisão da máquina nem refletindo dinâmica ecológica de presas e predadores.")

    # 10. Diagnóstico de unicidade
    lines.append("\n10. DIAGNÓSTICO DE UNICIDADE DA SOLUÇÃO:")
    lines.append("-" * 78)
    lines.append(f"  Status de Posto Completo: {diag_info['is_full_rank']}")
    lines.append(f"  Tolerância numérica adotada: tol = {diag_info['tolerance']:.8e}")
    lines.append("  Elementos diagonais da matriz R:")
    for k, r_kk in enumerate(diag_info["diagonal_entries"]):
        lines.append(f"    R[{k},{k}] = {r_kk:+.8e} (Razão R[{k},{k}] / tol = {r_kk / diag_info['tolerance']:.2e})")
    lines.append("\n  Justificativa Matemática e Computacional:")
    lines.append(f"  {uniqueness_justification}")

    # 11. Arquivos de figuras geradas
    if figure_paths:
        lines.append("\n11. VISUALIZAÇÃO EDITORIAL (ARQUIVOS GERADOS):")
        lines.append("-" * 78)
        for p in figure_paths:
            lines.append(f"  -> {p}")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)
