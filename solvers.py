"""Iterative solvers for sparse SPD linear systems."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np


@dataclass
class SolverResult:
    method: str
    x: np.ndarray
    iterations: int
    elapsed: float
    residual_relative: float
    converged: bool


def jacobi(A: Any, b: np.ndarray, tol: float, max_iter: int = 20000) -> SolverResult:
    _validate_for_iteration(A)
    diag = np.asarray(A.diagonal(), dtype=float)
    x = np.zeros_like(b, dtype=float)
    b_norm = _safe_norm(b)

    start_time = perf_counter()
    residual = b - A @ x
    residual_relative = np.linalg.norm(residual) / b_norm

    iterations = 0
    while residual_relative >= tol and iterations < max_iter:
        ax = A @ x
        x = x + (b - ax) / diag
        iterations += 1
        residual = b - A @ x
        residual_relative = np.linalg.norm(residual) / b_norm

    return SolverResult(
        method="Jacobi",
        x=x,
        iterations=iterations,
        elapsed=perf_counter() - start_time,
        residual_relative=residual_relative,
        converged=residual_relative < tol,
    )


def gauss_seidel(A: Any, b: np.ndarray, tol: float, max_iter: int = 20000) -> SolverResult:
    _validate_for_iteration(A)
    x = np.zeros_like(b, dtype=float)
    b_norm = _safe_norm(b)

    start_time = perf_counter()
    residual = b - A @ x
    residual_relative = np.linalg.norm(residual) / b_norm

    iterations = 0
    while residual_relative >= tol and iterations < max_iter:
        _gauss_seidel_step(A, b, x)
        iterations += 1
        residual = b - A @ x
        residual_relative = np.linalg.norm(residual) / b_norm

    return SolverResult(
        method="Gauss-Seidel",
        x=x,
        iterations=iterations,
        elapsed=perf_counter() - start_time,
        residual_relative=residual_relative,
        converged=residual_relative < tol,
    )


def gradient(A: Any, b: np.ndarray, tol: float, max_iter: int = 20000) -> SolverResult:
    _validate_for_iteration(A)
    x = np.zeros_like(b, dtype=float)
    b_norm = _safe_norm(b)

    start_time = perf_counter()
    residual = b - A @ x
    residual_relative = np.linalg.norm(residual) / b_norm

    iterations = 0
    while residual_relative >= tol and iterations < max_iter:
        Ar = A @ residual
        denom = float(np.dot(residual, Ar))
        if denom <= 0:
            break
        alpha = float(np.dot(residual, residual)) / denom
        x = x + alpha * residual
        residual = b - A @ x
        residual_relative = np.linalg.norm(residual) / b_norm
        iterations += 1

    return SolverResult(
        method="Gradiente",
        x=x,
        iterations=iterations,
        elapsed=perf_counter() - start_time,
        residual_relative=residual_relative,
        converged=residual_relative < tol,
    )


def conjugate_gradient(
    A: Any,
    b: np.ndarray,
    tol: float,
    max_iter: int = 20000,
) -> SolverResult:
    _validate_for_iteration(A)
    x = np.zeros_like(b, dtype=float)
    b_norm = _safe_norm(b)

    start_time = perf_counter()
    residual = b - A @ x
    direction = residual.copy()
    residual_norm_sq = float(np.dot(residual, residual))
    residual_relative = np.sqrt(residual_norm_sq) / b_norm

    iterations = 0
    while residual_relative >= tol and iterations < max_iter:
        Ad = A @ direction
        denom = float(np.dot(direction, Ad))
        if denom <= 0:
            break
        alpha = residual_norm_sq / denom
        x = x + alpha * direction
        residual = residual - alpha * Ad
        new_residual_norm_sq = float(np.dot(residual, residual))
        beta = new_residual_norm_sq / residual_norm_sq
        direction = residual + beta * direction
        residual_norm_sq = new_residual_norm_sq
        residual_relative = np.sqrt(residual_norm_sq) / b_norm
        iterations += 1

    return SolverResult(
        method="Gradiente coniugato",
        x=x,
        iterations=iterations,
        elapsed=perf_counter() - start_time,
        residual_relative=residual_relative,
        converged=residual_relative < tol,
    )


SOLVERS: tuple[Callable[[Any, np.ndarray, float, int], SolverResult], ...] = (
    jacobi,
    gauss_seidel,
    gradient,
    conjugate_gradient,
)


def _gauss_seidel_step(A: Any, b: np.ndarray, x: np.ndarray) -> None:
    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        diag = 0.0
        sigma = 0.0
        for idx in range(start, end):
            j = A.indices[idx]
            aij = A.data[idx]
            if j == i:
                diag = aij
            else:
                sigma += aij * x[j]
        if diag == 0:
            raise ValueError("La matrice ha almeno un elemento diagonale nullo")
        x[i] = (b[i] - sigma) / diag


def _validate_for_iteration(A: Any) -> None:
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice deve essere quadrata")
    diag = np.asarray(A.diagonal(), dtype=float)
    if np.any(diag == 0):
        raise ValueError("La matrice ha almeno un elemento diagonale nullo")


def _safe_norm(x: np.ndarray) -> float:
    norm = float(np.linalg.norm(x))
    if norm == 0:
        return 1.0
    return norm
