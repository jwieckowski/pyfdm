# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from scipy.optimize import minimize

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger
from ...validator import Validator

__all__ = ["fBWM"]

class fBWM(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Best-Worst Method (F-BWM).

    Fuzzy BWM extends Rezaei's Best-Worst Method to a fuzzy environment: the
    decision-maker provides a Best-to-Others vector (fuzzy preference of the
    best criterion over every other criterion) and an Others-to-Worst
    vector (fuzzy preference of every criterion over the worst one), both
    as Triangular Fuzzy Numbers (TFN). Fuzzy weights are obtained by
    minimizing the maximum deviation (k, the fuzzy consistency measure)
    between the ratio of derived weights and the corresponding stated
    preferences, subject to weight ordering (l <= m <= u) and normalization
    constraints. Since fuzzy division is only an approximation, the ratio
    constraints are linearized (each ``|a/b - c| <= k`` becomes a pair of
    linear inequalities obtained by multiplying through by the denominator)
    and solved as a nonlinear program with SLSQP.

    Attributes
    ----------
    k_ : float | None
        The optimal fuzzy consistency measure (deviation k) found by the
        optimizer, populated after `_calculate` runs.
    consistency : float | None
        Alias of `k_`, set via the inherited attribute for consistency with
        other subjective methods (`BaseSubjectiveFuzzyMethod`).

    Parameters
    ----------
    tol : float, default=1e-8
        Optimization tolerance (passed as `ftol` to the SLSQP optimizer).
    max_iter : int, default=1000
        Maximum optimizer iterations.
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.

    References
    ----------
    Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria
    decision-making method and its applications. Knowledge-Based
    Systems, 121, 23-31.

    Examples
    --------
    >>> from pyfdm.weights.subjective import fBWM
    >>> # 3 criteria, criterion 0 is best, criterion 2 is worst
    >>> best_to_others = [(1, 1, 1), (2/3, 1, 3/2), (3/2, 2, 5/2)]
    >>> others_to_worst = [(3/2, 2, 5/2), (2/3, 1, 3/2), (1, 1, 1)]
    >>> fbwm = fBWM()
    >>> weights = fbwm(
    ...     best_to_others=best_to_others,
    ...     others_to_worst=others_to_worst,
    ...     best_idx=0,
    ...     worst_idx=2,
    ... )
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 1000,
        logger: StepLogger | None = None,
    ):
        super().__init__(logger=logger)
        self.tol = tol
        self.max_iter = max_iter
        self.k_: float | None = None

    @staticmethod
    def _split(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Splits the flat optimization vector into its (l, m, u, k) parts.

        Parameters
        ----------
        x : np.ndarray
            Flat optimization vector of length ``3*n + 1``: the first `n`
            entries are the lower bounds (l), the next `n` are the modal
            values (m), the next `n` are the upper bounds (u), and the last
            entry is the consistency measure k.
        n : int
            Number of criteria.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
            ``(l, m, u, k)``, where `l`, `m`, `u` have shape (n,) and `k` is
            a scalar.
        """
        l = x[:n]
        m = x[n:2 * n]
        u = x[2 * n:3 * n]

        k = x[-1]

        return l, m, u, k

    @staticmethod
    def _initial_guess(n: int) -> np.ndarray:
        """
        Builds the initial optimization vector: equal crisp weights
        (1/n, 1/n, 1/n) for every criterion, and an initial consistency
        measure k of 1.0.

        Parameters
        ----------
        n : int
            Number of criteria.

        Returns
        -------
        np.ndarray
            Initial guess of length ``3*n + 1``.
        """
        return np.concatenate(
            [
                np.full(n, 1 / n),
                np.full(n, 1 / n),
                np.full(n, 1 / n),
                [1.0]
            ]
        )

    @staticmethod
    def _objective(x: np.ndarray) -> float:
        """
        Optimization objective: the consistency measure k (last entry of
        `x`), to be minimized.

        Parameters
        ----------
        x : np.ndarray
            Flat optimization vector, as produced/consumed by `_split`.

        Returns
        -------
        float
            The value of k.
        """
        return x[-1]

    def _calculate(
        self,
        best_to_others: list[tuple[float, float, float]] | np.ndarray,
        others_to_worst: list[tuple[float, float, float]] | np.ndarray,
        best_idx: int,
        worst_idx: int,
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the F-BWM method.

        Parameters
        ----------
        best_to_others : list[tuple[float, float, float]] | np.ndarray
            Fuzzy Best-to-Others vector, shape (n, 3), as (l, m, u) triples.
            The entry at `best_idx` is expected to be (1, 1, 1).
        others_to_worst : list[tuple[float, float, float]] | np.ndarray
            Fuzzy Others-to-Worst vector, shape (n, 3), as (l, m, u)
            triples. The entry at `worst_idx` is expected to be (1, 1, 1).
        best_idx : int
            Zero-based index of the best (most important) criterion.
        worst_idx : int
            Zero-based index of the worst (least important) criterion.

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n, 3), as (l, m, u) triples.

        Raises
        ------
        ValueError
            If `best_to_others`/`others_to_worst` cannot be converted to
            float arrays, or fail `Validator.validate_bwm_input`.
        RuntimeError
            If the SLSQP optimizer raises unexpectedly, or converges
            without success (`result.success` is False).
        """

        try:
            b2o = np.asarray(best_to_others, dtype=float)
            o2w = np.asarray(others_to_worst, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'best_to_others' and 'others_to_worst' must be array-like "
                f"numeric data convertible to float numpy arrays: {e}"
            ) from e

        try:
            Validator.validate_bwm_input(b2o, o2w, best_idx, worst_idx)
        except Exception as e:
            raise ValueError(f"Invalid F-BWM input: {e}") from e

        n = b2o.shape[0]

        self._log_step(self.logger, "Best-to-Others", b2o)
        self._log_step(self.logger, "Others-to-Worst", o2w)

        def constraints_fun(x: np.ndarray) -> np.ndarray:
            l, m, u, k = self._split(x, n)

            l_B = l[best_idx]
            m_B = m[best_idx]
            u_B = u[best_idx]

            l_W = l[worst_idx]
            m_W = m[worst_idx]
            u_W = u[worst_idx]

            cons = []
            # TFN ordering
            cons.append(m - l)
            cons.append(u - m)

            # Best-to-Others: |w_B / w_j - b2o_j| <= k, linearized
            cons.append((b2o[:, 0] + k) * u - l_B)
            cons.append(l_B - (b2o[:, 0] - k) * u)
            cons.append((b2o[:, 1] + k) * m - m_B)
            cons.append(m_B - (b2o[:, 1] - k) * m)
            cons.append((b2o[:, 2] + k) * l - u_B)
            cons.append(u_B - (b2o[:, 2] - k) * l)

            # Others-to-Worst: |w_j / w_W - o2w_j| <= k, linearized
            cons.append((o2w[:, 0] + k) * u_W - l)
            cons.append(l - (o2w[:, 0] - k) * u_W)
            cons.append((o2w[:, 1] + k) * m_W - m)
            cons.append(m - (o2w[:, 1] - k) * m_W)
            cons.append((o2w[:, 2] + k) * l_W - u)
            cons.append(u - (o2w[:, 2] - k) * l_W)

            return np.concatenate(cons)

        x0 = self._initial_guess(n)

        constraints = [
            {
                "type": "eq",
                "fun": lambda x:
                    np.sum(
                        x[:n]
                        + 4 * x[n:2 * n]
                        + x[2 * n:3 * n]
                    ) - 6
            },
            {
                "type": "ineq",
                "fun": constraints_fun
            }
        ]

        bounds = [(0.0, None)] * (3 * n + 1)

        try:
            result = minimize(
                self._objective,
                x0,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tol,
                }
            )
        except Exception as e:
            raise RuntimeError(f"fBWM optimization raised unexpectedly: {e}") from e

        if not result.success:
            raise RuntimeError(f"fBWM optimization failed: {result.message}")

        l_opt, m_opt, u_opt, k_opt = self._split(result.x, n)
        self.k_ = k_opt
        self.consistency = k_opt

        self._log_step(self.logger, "Optimization error k", k_opt)

        weights = np.column_stack((l_opt, m_opt, u_opt))

        self._log_step(self.logger, "Raw fuzzy weights", weights)

        return weights