"""CLI runner for Problem 2 of Computational Work 3 (TC3).

Executes Problem 2 securely:
- Prompts for student enrollment number at runtime using getpass (without terminal echoing).
- Generates experimental data vectors x and y (predator-prey observations).
- Constructs the Vandermonde design matrix X.
- Solves the least-squares problem via reduced QR factorization (Modified Gram-Schmidt).
- Evaluates the Mean Squared Error (MSE), residuals, and verification indicators.
- Produces an editorial-quality figure adhering to Edward Tufte principles.
- Exports publication-ready PDF and PNG figures to local ignored directories.
- Displays the complete mathematical and numerical report on the terminal.

Usage:
    python solve_problem2.py
"""

from __future__ import annotations

import os
import sys

# Ensure repository root and src directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

for path in [REPO_ROOT, SRC_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from matrix_methods.linear_algebra import RankDeficiencyError
from matrix_methods.problem1 import read_and_validate_matricula
from matrix_methods.problem2 import (
    format_problem2_report,
    plot_problem2_fit,
    run_problem2,
)


def main() -> None:
    # Ensure proper Unicode display in Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 78)
    print("MÉTODOS MATRICIAIS COMPUTACIONAIS — TRABALHO COMPUTACIONAL 3 (TC3)")
    print("PROBLEMA 2: AJUSTE POLINOMIAL QUADRÁTICO (MÍNIMOS QUADRADOS)")
    print("=" * 78)
    print("Aviso de Privacidade: A matrícula é solicitada de forma segura sem eco no terminal.")
    print("Nenhum dado pessoal ou resultado será gravado em disco ou versionado.")
    print("-" * 78)

    try:
        m_processed = read_and_validate_matricula()
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
        sys.exit(1)
    except ValueError as err:
        print(f"\nErro de validação: {err}")
        sys.exit(1)

    print("\nExecutando ajuste quadrático via Gram-Schmidt Modificado...")

    try:
        results = run_problem2(m_processed)
    except RankDeficiencyError as r_err:
        print(f"\n[DIAGNÓSTICO CRÍTICO DE POSTO] {r_err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nErro inesperado durante a execução: {exc}")
        sys.exit(1)

    # Configure single canonical local export path (strictly ignored by .gitignore)
    local_results_dir = os.path.join(CURRENT_DIR, "local_results")
    os.makedirs(local_results_dir, exist_ok=True)

    fig_pdf = os.path.join(local_results_dir, "problem2_quadratic_fit.pdf")
    fig_png = os.path.join(local_results_dir, "problem2_quadratic_fit.png")

    target_fig_paths = [fig_pdf, fig_png]

    # Generate editorial visualization figure
    generated_figures = plot_problem2_fit(
        x_points=results["x"],
        y_points=results["y"],
        theta=results["theta"],
        mse=results["mse"],
        output_paths=target_fig_paths,
    )

    # Format and display analytical report
    report = format_problem2_report(results, figure_paths=generated_figures)
    print("\n" + report)

    # Save text report to local results directory (ignored by .gitignore)
    results_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_file = os.path.join(results_dir, "tc3_problem2_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + "=" * 78)
    print(f"[ARQUIVO LOCAL] Relatório salvo com sucesso em: {report_file}")
    print("[SEGURANÇA] O relatório e as figuras estão ignorados pelo .gitignore e não serão versionados.")
    print("=" * 78)


if __name__ == "__main__":
    main()
