"""CLI runner for Problem 1 of Computational Work 3 (TC3).

Executes Problem 1 securely: prompts for the student enrollment number at runtime
using getpass, performs QR factorization via Modified Gram-Schmidt, calculates
the least-squares solution, evaluates verification indicators, and displays the
complete numerical report on the terminal.

Usage:
    python solve_problem1.py
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

for path in [REPO_ROOT, SRC_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from matrix_methods.linear_algebra import RankDeficiencyError
from matrix_methods.problem1 import (
    format_results_report,
    read_and_validate_matricula,
    run_problem1,
)


def main() -> None:
    # Ensure proper Unicode display in Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 78)
    print("MÉTODOS MATRICIAIS COMPUTACIONAIS — TRABALHO COMPUTACIONAL 3 (TC3)")
    print("PROBLEMA 1: FATORAÇÃO QR E SISTEMA DE MÍNIMOS QUADRADOS")
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

    print("\nExecutando fatoração QR por Gram-Schmidt Modificado e resolvendo o sistema...")

    try:
        results = run_problem1(m_processed)
    except RankDeficiencyError as r_err:
        print(f"\n[DIAGNÓSTICO CRÍTICO DE POSTO] {r_err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nErro inesperado durante a execução: {exc}")
        sys.exit(1)

    report = format_results_report(results)
    print("\n" + report)

    # Save to local results/ directory inside Computational_Work_3 (explicitly ignored by .gitignore)
    results_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_file_unit = os.path.join(results_dir, "tc3_problem1_report.txt")
    with open(report_file_unit, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + "=" * 78)
    print(f"[ARQUIVO LOCAL] Relatório salvo com sucesso em: {report_file_unit}")
    print("[SEGURANÇA] O arquivo 'tc3_problem1_report.txt' está ignorado pelo .gitignore e não será versionado.")
    print("=" * 78)


if __name__ == "__main__":
    main()
