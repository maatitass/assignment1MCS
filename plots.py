from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

from solvers import SOLVERS, warmup

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dati"

# Ordine e colori dei metodi, coerenti in tutti i grafici di approfondimento.
METHOD_ORDER = [solver.name for solver in SOLVERS]
METHOD_COLORS = {
    "Jacobi": "tab:blue",
    "Gauss-Seidel": "tab:orange",
    "Gradiente": "tab:green",
    "Gradiente coniugato": "tab:red",
}


# ---------------------------------------------------------------------------
# Funzioni di servizio condivise.
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    """Layout, salvataggio a 200 dpi e chiusura della figura."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Risultati non trovati: {csv_path}. Esegui prima 'python run_assignment.py'.")
    with csv_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_cond(cond_csv: Path) -> dict[str, float]:
    if not cond_csv.exists():
        raise FileNotFoundError(
            f"File dei condizionamenti non trovato: {cond_csv}. "
            "Esegui prima 'python analyze_cond.py'.")
    with cond_csv.open(encoding="utf-8", newline="") as file:
        return {row["matrix"]: float(row["cond"]) for row in csv.DictReader(file)}


def _closest_tol(rows: list[dict[str, str]], target: float) -> str:
    """Etichetta di tolleranza presente nel CSV piu' vicina a ``target``."""
    tols = sorted({row["tol"] for row in rows}, key=float)
    return min(tols, key=lambda t: abs(np.log10(float(t)) - np.log10(target)))


def _methods_present(rows: list[dict[str, str]]) -> list[str]:
    present = {row["method"] for row in rows}
    return [m for m in METHOD_ORDER if m in present]


def _grouped_bars(ax, groups, series, value_fn) -> None:
    """Disegna barre raggruppate: un gruppo per ``groups``, una barra per
    ``series``. ``value_fn(group, serie)`` ritorna l'altezza (o None)."""
    n_series = max(len(series), 1)
    width = 0.8 / n_series
    x = np.arange(len(groups))
    for j, serie in enumerate(series):
        offset = (j - (n_series - 1) / 2) * width
        heights = [value_fn(group, serie) or np.nan for group in groups]
        ax.bar(x + offset, heights, width=width, label=serie,
               color=METHOD_COLORS.get(serie))


# ---------------------------------------------------------------------------
# Grafici fondamentali: metrica vs tolleranza (richiamati da run_assignment.py).
# ---------------------------------------------------------------------------
def create_plots(rows: list[dict[str, str]], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plots_dir / "relative_error.png",
        plots_dir / "iterations.png",
        plots_dir / "time.png",
    ]
    _plot_metric(rows, "relative_error", "Errore relativo", paths[0], log_y=True)
    _plot_metric(rows, "iterations", "Iterazioni", paths[1], log_y=False)
    _plot_metric(rows, "elapsed_seconds", "Tempo [s]", paths[2], log_y=True)
    return paths


def _plot_metric(
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    path: Path,
    log_y: bool,
) -> None:
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["matrix"], row["method"])].append(row)

    matrices = sorted({row["matrix"] for row in rows})
    fig, axes = plt.subplots(len(matrices), 1, figsize=(9, 3.2 * len(matrices)), squeeze=False)

    for ax, matrix_name in zip(axes[:, 0], matrices):
        for (current_matrix, method), values in grouped.items():
            if current_matrix != matrix_name:
                continue
            values = sorted(values, key=lambda item: float(item["tol"]))
            xs = [float(item["tol"]) for item in values]
            ys = [float(item[metric]) for item in values]
            ax.plot(xs, ys, marker="o", label=method)

        ax.set_title(matrix_name)
        ax.set_xlabel("Tolleranza")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.invert_xaxis()
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    _save(fig, path)


# ---------------------------------------------------------------------------
# A) Storia di convergenza.
# ---------------------------------------------------------------------------
def compute_histories(tol: float, max_iter: int) -> dict[str, dict[str, list[float]]]:
    """Riesegue i solutori registrando la storia del residuo scalato.

    Ritorna  histories[matrice][metodo] = [r0, r1, ...]  (r0 = 1).
    """
    warmup()
    paths = sorted(DATA_DIR.glob("*.mtx"))
    histories: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        A = csr_matrix(mmread(path), dtype=float)
        # .shape e' tipato Optional negli stub di scipy: a runtime e' sempre una
        # tupla (n, n), quindi il pedice e' sicuro.
        x_exact = np.ones(A.shape[0], dtype=float)  # type: ignore[reportOptionalSubscript]
        b = A @ x_exact
        per_method: dict[str, list[float]] = {}
        for solver in SOLVERS:
            result = solver.solve(A, b, tol, max_iter, record_history=True)
            per_method[solver.name] = result.history or []
        histories[path.name] = per_method
        print(f"  storia calcolata: {path.name}")
    return histories


def plot_convergence(histories: dict[str, dict[str, list[float]]], tol: float,
                     path: Path) -> None:
    import matplotlib.pyplot as plt

    matrices = sorted(histories)
    fig, axes = plt.subplots(len(matrices), 1, figsize=(9, 3.2 * len(matrices)),
                             squeeze=False)
    for ax, matrix_name in zip(axes[:, 0], matrices):
        for method in METHOD_ORDER:
            history = histories[matrix_name].get(method)
            if not history or len(history) < 2:
                continue
            # Saltiamo il punto iniziale (r0 = 1, k = 0) perche' l'asse x e' in
            # scala logaritmica: cosi' tutti i metodi sono visibili anche quando
            # il numero di iterazioni varia di ordini di grandezza (es. CG ~10^2
            # vs Gradiente ~10^4 sulle matrici spa).
            tail = history[1:]
            ax.plot(range(1, len(tail) + 1), tail, label=method,
                    color=METHOD_COLORS.get(method))
        ax.axhline(tol, color="gray", linestyle="--", linewidth=1,
                   label=f"tol = {tol:g}")
        ax.set_title(matrix_name)
        ax.set_xlabel("Iterazione k")
        ax.set_ylabel(r"$\|r_k\| / \|b\|$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# B) Iterazioni vs cond(A).
# ---------------------------------------------------------------------------
def plot_iter_vs_cond(rows: list[dict[str, str]], cond_map: dict[str, float],
                      tol_label: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["tol"] == tol_label]
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in _methods_present(selected):
        points = [(cond_map[row["matrix"]], int(row["iterations"]))
                  for row in selected
                  if row["method"] == method and row["matrix"] in cond_map]
        points.sort()
        if not points:
            continue
        conds = [p[0] for p in points]
        iters = [p[1] for p in points]
        ax.plot(conds, iters, marker="o", label=method,
                color=METHOD_COLORS.get(method))

    # Curve teoriche di riferimento, ancorate al punto a cond minimo del
    # Gradiente / CG (numero di iterazioni ~ cond per il Gradiente, ~sqrt(cond)
    # per il CG).
    conds_axis = np.array(sorted(cond_map.values()))
    _add_reference_curve(ax, selected, cond_map, "Gradiente", conds_axis,
                         lambda c, c0: c / c0, r"$\propto \kappa(A)$")
    _add_reference_curve(ax, selected, cond_map, "Gradiente coniugato",
                         conds_axis, lambda c, c0: np.sqrt(c / c0),
                         r"$\propto \sqrt{\kappa(A)}$")

    ax.set_title(f"Iterazioni vs numero di condizionamento (tol = {tol_label})")
    ax.set_xlabel(r"$\kappa(A) = \lambda_{max} / \lambda_{min}$")
    ax.set_ylabel("Iterazioni")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)


def _add_reference_curve(ax, selected, cond_map, method, conds_axis,
                         shape, label) -> None:
    """Disegna una curva teorica ancorata al dato reale a cond minimo."""
    points = [(cond_map[row["matrix"]], int(row["iterations"]))
              for row in selected
              if row["method"] == method and row["matrix"] in cond_map]
    if not points:
        return
    points.sort()
    c0, it0 = points[0]
    ys = [it0 * shape(c, c0) for c in conds_axis]
    ax.plot(conds_axis, ys, linestyle=":", color=METHOD_COLORS.get(method),
            alpha=0.7, label=f"{method}: {label}")


# ---------------------------------------------------------------------------
# C) Amplificazione errore/residuo.
# ---------------------------------------------------------------------------
def plot_error_vs_residual(rows: list[dict[str, str]], cond_map: dict[str, float],
                           tol_label: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["tol"] == tol_label]
    matrices = sorted({row["matrix"] for row in selected})
    methods = _methods_present(selected)

    ratios = defaultdict(dict)  # ratios[matrix][method] = err/res
    for row in selected:
        res = float(row["residual_relative"])
        if res > 0:
            ratios[row["matrix"]][row["method"]] = float(row["relative_error"]) / res

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_bars(ax, matrices, methods,
                  lambda mat, met: ratios.get(mat, {}).get(met))

    # Riferimento: il rapporto err/res e' limitato superiormente da cond(A).
    for i, matrix_name in enumerate(matrices):
        if matrix_name in cond_map:
            ax.hlines(cond_map[matrix_name], i - 0.4, i + 0.4,
                      color="black", linestyle="--", linewidth=1.2,
                      label="cond(A)" if i == 0 else "_nolegend_")

    ax.set_title(f"Amplificazione errore/residuo (tol = {tol_label})")
    ax.set_xlabel("Matrice")
    ax.set_ylabel(r"$\frac{\|x-x^*\|/\|x^*\|}{\|r\|/\|b\|}$  (≈ fattore di amplificazione)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels(matrices)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# D) Costo per iterazione + bar chart riassuntivi.
# ---------------------------------------------------------------------------
def plot_cost_per_iter(rows: list[dict[str, str]], tol_label: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["tol"] == tol_label]
    matrices = sorted({row["matrix"] for row in selected})
    methods = _methods_present(selected)

    cost = defaultdict(dict)  # cost[matrix][method] = elapsed/iterations
    for row in selected:
        iters = int(row["iterations"])
        if iters > 0:
            cost[row["matrix"]][row["method"]] = float(row["elapsed_seconds"]) / iters

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_bars(ax, matrices, methods,
                  lambda mat, met: cost.get(mat, {}).get(met))
    ax.set_title(f"Costo medio per iterazione (tol = {tol_label})")
    ax.set_xlabel("Matrice")
    ax.set_ylabel("Tempo per iterazione [s]")
    ax.set_yscale("log")
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels(matrices)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_summary_bars(rows: list[dict[str, str]], tol_label: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["tol"] == tol_label]
    matrices = sorted({row["matrix"] for row in selected})
    methods = _methods_present(selected)

    time_of = defaultdict(dict)
    iter_of = defaultdict(dict)
    for row in selected:
        time_of[row["matrix"]][row["method"]] = float(row["elapsed_seconds"])
        iter_of[row["matrix"]][row["method"]] = int(row["iterations"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    _grouped_bars(axes[0], matrices, methods,
                  lambda mat, met: time_of.get(mat, {}).get(met))
    axes[0].set_title(f"Tempo di calcolo (tol = {tol_label})")
    axes[0].set_ylabel("Tempo [s]")
    axes[0].set_yscale("log")

    _grouped_bars(axes[1], matrices, methods,
                  lambda mat, met: iter_of.get(mat, {}).get(met))
    axes[1].set_title(f"Iterazioni (tol = {tol_label})")
    axes[1].set_ylabel("Iterazioni")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("Matrice")
        ax.set_xticks(range(len(matrices)))
        ax.set_xticklabels(matrices)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Main: genera i grafici di approfondimento dai CSV (e dalla riesecuzione dei
# solutori per la storia di convergenza).
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(output_dir / "results.csv")
    cond_map = load_cond(output_dir / "cond.csv")

    bar_tol = _closest_tol(rows, args.bar_tol)

    print("Genero i grafici dal CSV...")
    plot_iter_vs_cond(rows, cond_map, bar_tol, plots_dir / "iter_vs_cond.png")
    plot_error_vs_residual(rows, cond_map, bar_tol, plots_dir / "error_vs_residual.png")
    plot_cost_per_iter(rows, bar_tol, plots_dir / "cost_per_iter.png")
    plot_summary_bars(rows, bar_tol, plots_dir / "summary_bars.png")

    print("Calcolo la storia di convergenza (rieseguo i solutori)...")
    histories = compute_histories(args.conv_tol, args.max_iter)
    plot_convergence(histories, args.conv_tol, plots_dir / "convergence.png")

    print(f"\nGrafici scritti in: {plots_dir}")
    for name in ("convergence.png", "iter_vs_cond.png", "error_vs_residual.png",
                 "cost_per_iter.png", "summary_bars.png"):
        print(f"  - {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafici di approfondimento (MCS Assignment 1)")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Cartella con results.csv/cond.csv e output plots/, default results.")
    parser.add_argument("--conv-tol", type=float, default=1e-10,
                        help="Tolleranza per la storia di convergenza, default 1e-10.")
    parser.add_argument("--bar-tol", type=float, default=1e-8,
                        help="Tolleranza per i grafici a barre / cond, default 1e-8.")
    parser.add_argument("--max-iter", type=int, default=20000,
                        help="Numero massimo di iterazioni, default 20000.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
