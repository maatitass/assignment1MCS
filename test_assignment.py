"""Small tests for the assignment code."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import mmread
from scipy.sparse import diags

from solvers import SOLVERS, ConjugateGradient


def test_scipy_reads_matrix_market() -> None:
    text = """%%MatrixMarket matrix coordinate real general
% tiny test matrix
3 3 5
1 1 2.0
1 2 -1.0
2 1 -1.0
2 2 2.0
3 3 3.0
"""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.mtx"
        path.write_text(text, encoding="utf-8")
        A = mmread(path).tocsr()

    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(A @ x, np.array([0.0, 3.0, 9.0]))
    assert np.allclose(A.diagonal(), np.array([2.0, 2.0, 3.0]))


def test_solvers_on_tridiagonal_spd() -> None:
    A = _tridiagonal_spd(20)
    x_exact = np.ones(20)
    b = A @ x_exact

    for solver in SOLVERS:
        result = solver.solve(A, b, tol=1e-8, max_iter=20000)
        relative_error = np.linalg.norm(x_exact - result.x) / np.linalg.norm(x_exact)
        assert result.converged, solver.name
        assert relative_error < 1e-6, solver.name


def test_conjugate_gradient_is_fast_on_small_matrix() -> None:
    A = _tridiagonal_spd(20)
    x_exact = np.ones(20)
    b = A @ x_exact
    result = ConjugateGradient().solve(A, b, tol=1e-10, max_iter=20000)
    assert result.converged
    assert result.iterations <= 20


def _tridiagonal_spd(n: int):
    return diags(
        diagonals=[-np.ones(n - 1), 3 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )


if __name__ == "__main__":
    test_scipy_reads_matrix_market()
    test_solvers_on_tridiagonal_spd()
    test_conjugate_gradient_is_fast_on_small_matrix()
    print("Tutti i test sono passati.")
