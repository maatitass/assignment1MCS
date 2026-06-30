from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Risultato di una risoluzione.
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    method: str            # nome del metodo
    x: np.ndarray          # soluzione approssimata x^(k)
    iterations: int        # numero di iterazioni eseguite
    elapsed: float         # tempo di calcolo in secondi
    residual_relative: float  # ||b - A x||/||b|| raggiunto
    converged: bool        # True se il criterio di arresto e' stato soddisfatto
    history: list[float] | None = None  # storia del residuo scalato (solo se richiesta)


# ---------------------------------------------------------------------------
# Iterazione astratta + classe base con lo scheletro comune.
# ---------------------------------------------------------------------------
class _Iteration(ABC):
    """Stato interno di un metodo iterativo per un singolo problema.

    Mantiene (e aggiorna in place) il vettore soluzione ``x`` e quanto serve per
    calcolare il residuo ed eseguire il passo di aggiornamento.
    """

    @abstractmethod
    def residual_norm(self) -> float:
        """Norma euclidea del residuo corrente ||b - A x^(k)||."""

    @abstractmethod
    def step(self) -> None:
        """Aggiorna lo stato: x^(k) -> x^(k+1)."""


class IterativeSolver(ABC):
    """Classe base: implementa lo scheletro comune a tutti i metodi."""

    name: str = "iterative"
    key: str = "iterative"          # identificativo per la CLI
    DEFAULT_MAX_ITER: int = 20000

    @abstractmethod
    def _make_iteration(self, A: Any, b: np.ndarray, x: np.ndarray) -> _Iteration:
        """Costruisce l'oggetto iterazione specifico del metodo."""

    def solve(self, A: Any, b: np.ndarray, tol: float,
              max_iter: int | None = None,
              record_history: bool = False) -> SolverResult:
        """Risolve Ax = b partendo da x^(0) = 0.

        Si arresta quando il residuo scalato scende sotto ``tol`` oppure quando
        si superano ``max_iter`` iterazioni (in tal caso converged=False).

        Se ``record_history=True`` salva in ``SolverResult.history`` il residuo
        scalato a ogni iterazione (partendo dal valore iniziale = 1). Il valore
        e' gia' calcolato dal ciclo, quindi non comporta matvec aggiuntivi; va
        comunque usato SOLO per i grafici di convergenza, NON per le misure di
        tempo (l'append alla lista introduce un piccolo overhead di Python).
        """
        _validate_for_iteration(A)
        if max_iter is None:
            max_iter = self.DEFAULT_MAX_ITER

        b = np.asarray(b, dtype=float)
        x = np.zeros_like(b, dtype=float)          # vettore iniziale nullo
        b_norm = _safe_norm(b)

        start_time = perf_counter()
        it = self._make_iteration(A, b, x)
        iterations = 0
        residual_relative = it.residual_norm() / b_norm   # all'inizio vale 1
        history = [residual_relative] if record_history else None
        while residual_relative >= tol and iterations < max_iter:
            it.step()
            iterations += 1
            residual_relative = it.residual_norm() / b_norm
            if history is not None:
                history.append(residual_relative)
        elapsed = perf_counter() - start_time

        return SolverResult(
            method=self.name,
            x=x.copy(),
            iterations=iterations,
            elapsed=elapsed,
            residual_relative=residual_relative,
            converged=residual_relative < tol,
            history=history,
        )


# ---------------------------------------------------------------------------
# 1) Metodo di Jacobi.   P = diag(A),  x <- x + P^{-1} r,   r = b - A x.
# ---------------------------------------------------------------------------
class _JacobiIteration(_Iteration):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        self.inv_diag = 1.0 / np.asarray(A.diagonal(), dtype=float)  # P^{-1}
        self.residual = b - A @ x                  # residuo iniziale (= b)

    def residual_norm(self):
        return float(np.linalg.norm(self.residual))

    def step(self):
        # x^(k+1) = x^(k) + P^{-1} r^(k)
        self.x += self.inv_diag * self.residual
        self.residual = self.b - self.A @ self.x


class Jacobi(IterativeSolver):
    name = "Jacobi"
    key = "jacobi"

    def _make_iteration(self, A, b, x):
        return _JacobiIteration(A, b, x)


# ---------------------------------------------------------------------------
# 2) Metodo di Gauss-Seidel.  P = parte triangolare inferiore di A.
#    sweep in place (= sostituzione in avanti),  poi  r = b - A x.
# ---------------------------------------------------------------------------
class _GaussSeidelIteration(_Iteration):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        self.diag = np.asarray(A.diagonal(), dtype=float)
        self.indptr = A.indptr
        self.indices = A.indices
        self.data = np.asarray(A.data, dtype=float)
        self.residual = b - A @ x

    def residual_norm(self):
        return float(np.linalg.norm(self.residual))

    def step(self):
        _gauss_seidel_sweep(self.indptr, self.indices, self.data,
                            self.diag, self.b, self.x)
        self.residual = self.b - self.A @ self.x


class GaussSeidel(IterativeSolver):
    name = "Gauss-Seidel"
    key = "gauss-seidel"

    def _make_iteration(self, A, b, x):
        return _GaussSeidelIteration(A, b, x)


# ---------------------------------------------------------------------------
# 3) Metodo del Gradiente (steepest descent).
#    alpha = (r·r)/(r·A r) ;  x <- x + alpha r ;  r <- r - alpha A r.
# ---------------------------------------------------------------------------
class _GradientIteration(_Iteration):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        self.residual = b - A @ x                   # = -grad(phi) in x

    def residual_norm(self):
        return float(np.linalg.norm(self.residual))

    def step(self):
        Ar = self.A @ self.residual
        denom = float(np.dot(self.residual, Ar))
        alpha = float(np.dot(self.residual, self.residual)) / denom
        self.x += alpha * self.residual
        # r^(k+1) = b - A x^(k+1) = r^(k) - alpha A r^(k): niente matvec extra
        self.residual = self.residual - alpha * Ar


class Gradient(IterativeSolver):
    name = "Gradiente"
    key = "gradient"

    def _make_iteration(self, A, b, x):
        return _GradientIteration(A, b, x)


# ---------------------------------------------------------------------------
# 4) Metodo del Gradiente coniugato.
#    Direzioni A-coniugate: niente "zig-zag", convergenza in <= n passi.
# ---------------------------------------------------------------------------
class _ConjugateGradientIteration(_Iteration):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        self.residual = b - A @ x                   # residuo iniziale (= b)
        self.direction = self.residual.copy()       # d^(0) = r^(0)
        self.residual_norm_sq = float(np.dot(self.residual, self.residual))

    def residual_norm(self):
        return float(np.sqrt(self.residual_norm_sq))

    def step(self):
        Ad = self.A @ self.direction
        denom = float(np.dot(self.direction, Ad))
        alpha = self.residual_norm_sq / denom
        self.x += alpha * self.direction
        self.residual = self.residual - alpha * Ad
        new_norm_sq = float(np.dot(self.residual, self.residual))
        beta = new_norm_sq / self.residual_norm_sq
        self.direction = self.residual + beta * self.direction
        self.residual_norm_sq = new_norm_sq


class ConjugateGradient(IterativeSolver):
    name = "Gradiente coniugato"
    key = "conjugate-gradient"

    def _make_iteration(self, A, b, x):
        return _ConjugateGradientIteration(A, b, x)


# Istanze dei quattro metodi, nell'ordine richiesto dalla consegna.
SOLVERS: tuple[IterativeSolver, ...] = (
    Jacobi(),
    GaussSeidel(),
    Gradient(),
    ConjugateGradient(),
)


# ---------------------------------------------------------------------------
# Routine compilata (numba) e funzioni di servizio condivise.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _gauss_seidel_sweep(indptr, indices, data, diag, b, x):
    """Una iterazione (sweep) di Gauss-Seidel, aggiornando x in place.

    Equivale a risolvere (D + L) y = r per sostituzione in avanti: ogni x[i] usa
    i valori gia' aggiornati x[0..i-1] e quelli vecchi x[i+1..n-1]. E'
    un'operazione sequenziale (x[i] dipende dai precedenti) e quindi non
    vettorizzabile: la compiliamo con numba per renderla veloce restando codice
    nostro (non usiamo nessun solutore di libreria).
    """
    n = x.shape[0]
    for i in range(n):
        sigma = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i:
                sigma += data[k] * x[j]
        x[i] = (b[i] - sigma) / diag[i]


def warmup() -> None:
    """Forza la compilazione JIT (numba) su un problemino 3x3.

    Va chiamata PRIMA delle misure di tempo: altrimenti la prima esecuzione di
    Gauss-Seidel paga il costo (una tantum) di compilazione della sweep,
    falsando il tempo riportato.
    """
    from scipy.sparse import csr_matrix

    A = csr_matrix(np.array([[4.0, 1.0, 0.0],
                             [1.0, 4.0, 1.0],
                             [0.0, 1.0, 4.0]]))
    b = A @ np.ones(3)
    for solver in SOLVERS:
        solver.solve(A, b, 1e-12, 100)


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
