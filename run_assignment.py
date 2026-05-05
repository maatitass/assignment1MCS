"""Command line runner for the first MCS assignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import mmread

from plot_results import create_plots
from solvers import SOLVERS


DEFAULT_TOLS = (1e-4, 1e-6, 1e-8, 1e-10)
DEFAULT_MAX_ITER = 20000
METHOD_NAMES = tuple(solver.__name__.replace("_", "-") for solver in SOLVERS)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_paths = _resolve_matrices(args, base_dir)
    tolerances = args.tol if args.tol else list(DEFAULT_TOLS)
    solvers = _select_solvers(args.method)

    rows: list[dict[str, str]] = []
    for matrix_path in matrix_paths:
        print(f"\nMatrice: {matrix_path}")
        A = mmread(matrix_path).tocsr().astype(float)
        x_exact = np.ones(A.shape[0], dtype=float)
        b = A @ x_exact

        for tol in tolerances:
            print(f"  tol = {tol:g}")
            for solver in solvers:
                result = solver(A, b, tol, args.max_iter)
                relative_error = np.linalg.norm(x_exact - result.x) / np.linalg.norm(x_exact)
                row = {
                    "matrix": matrix_path.name,
                    "n": str(A.shape[0]),
                    "nnz": str(A.nnz),
                    "tol": f"{tol:.0e}",
                    "method": result.method,
                    "relative_error": f"{relative_error:.16e}",
                    "residual_relative": f"{result.residual_relative:.16e}",
                    "iterations": str(result.iterations),
                    "elapsed_seconds": f"{result.elapsed:.6f}",
                    "converged": str(result.converged),
                }
                rows.append(row)
                print(
                    "    "
                    f"{result.method:20s} "
                    f"it={result.iterations:6d} "
                    f"err={relative_error:.3e} "
                    f"res={result.residual_relative:.3e} "
                    f"time={result.elapsed:.3f}s "
                    f"conv={result.converged}"
                )

    csv_path = output_dir / "results.csv"
    _write_csv(rows, csv_path)
    print(f"\nCSV scritto in: {csv_path}")

    if args.plots:
        plot_paths = create_plots(rows, output_dir)
        print(f"Grafici scritti in: {output_dir / 'plots'}")
        for plot_path in plot_paths:
            print(f"  - {plot_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini libreria per sistemi lineari SPD")
    parser.add_argument(
        "--matrix",
        action="append",
        type=Path,
        help="Matrice .mtx da usare. Ripetibile. Se assente usa tutti i file .mtx in dati.",
    )
    parser.add_argument(
        "--tol",
        action="append",
        type=float,
        help="Tolleranza da usare. Ripetibile. Se assente usa 1e-4, 1e-6, 1e-8, 1e-10.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Numero massimo di iterazioni, default 20000.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=METHOD_NAMES,
        help=(
            "Metodo da usare. Ripetibile. Se assente usa tutti i metodi. "
            f"Valori: {', '.join(METHOD_NAMES)}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Cartella di output, default assignment 1/results.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Non generare grafici.",
    )
    parser.set_defaults(plots=True)
    return parser.parse_args()


def _resolve_matrices(args: argparse.Namespace, base_dir: Path) -> list[Path]:
    if args.matrix:
        paths = args.matrix
    else:
        data_dir = base_dir / "dati"
        paths = sorted(data_dir.glob("*.mtx"))
        if not paths:
            raise FileNotFoundError(f"Nessuna matrice .mtx trovata in: {data_dir}")

    resolved = []
    for path in paths:
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Matrice non trovata: {path}")
        resolved.append(path)
    return resolved


def _select_solvers(methods: list[str] | None):
    if not methods:
        return SOLVERS

    selected_names = {method.replace("-", "_") for method in methods}
    return tuple(solver for solver in SOLVERS if solver.__name__ in selected_names)


def _write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    fieldnames = [
        "matrix",
        "n",
        "nnz",
        "tol",
        "method",
        "relative_error",
        "residual_relative",
        "iterations",
        "elapsed_seconds",
        "converged",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
