"""Stima del numero di condizionamento delle matrici di test.

NOTA: questo script serve SOLO all'analisi dei risultati nella relazione
(spiegare perche' alcuni metodi sono piu' lenti). NON fa parte della mini
libreria di solutori: qui usiamo scipy.sparse.linalg.eigsh, che e' un
calcolatore di autovalori, per stimare lambda_max e lambda_min e quindi
cond(A) = lambda_max / lambda_min.

Uso:
    python analyze_cond.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from scipy.io import mmread
from scipy.sparse.linalg import eigsh

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dati"
OUTPUT_DIR = BASE_DIR / "results"


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.mtx"))
    print(f"{'matrice':<12s}{'n':>7s}{'lambda_min':>14s}{'lambda_max':>14s}{'cond(A)':>14s}")
    print("-" * 61)
    rows: list[dict[str, object]] = []
    for path in paths:
        A = mmread(path).tocsr().astype(float)
        n = A.shape[0]
        lam_max = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
        # piu' piccolo autovalore via shift-invert attorno a 0 (A SPD => lambda_min > 0)
        lam_min = float(eigsh(A, k=1, sigma=0.0, which="LM",
                              return_eigenvectors=False)[0])
        cond = lam_max / lam_min
        print(f"{path.name:<12s}{n:>7d}{lam_min:>14.3e}{lam_max:>14.3e}{cond:>14.3e}")
        rows.append({
            "matrix": path.name,
            "n": n,
            "lambda_min": f"{lam_min:.16e}",
            "lambda_max": f"{lam_max:.16e}",
            "cond": f"{cond:.16e}",
        })

    # Salviamo i valori anche su CSV: servono ai grafici (plots.py) e ad
    # arricchire le tabelle della relazione.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cond_csv = OUTPUT_DIR / "cond.csv"
    with cond_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["matrix", "n", "lambda_min", "lambda_max", "cond"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV scritto in: {cond_csv}")


if __name__ == "__main__":
    main()
