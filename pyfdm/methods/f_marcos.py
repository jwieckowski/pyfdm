# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger

from ..utils import graded_mean_average_defuzzification

class fMARCOS(BaseFuzzyMethod):
    """
    Fuzzy MARCOS — Measurement of Alternatives and Ranking according
    to Compromise Solution.

    The decision matrix is extended with an Anti-Ideal (AAI: worst values
    per criterion) and Ideal (AI: best values per criterion) row,
    normalized, and weighted. Each alternative's row sum (S_i) of the
    weighted matrix is compared against the anti-ideal's sum (S_aai) and
    the ideal's sum (S_ai) to obtain two utility degrees: K-i = S_i/S_aai
    (relative to the anti-ideal) and K+i = S_i/S_ai (relative to the
    ideal). These are combined into utility functions f(K-i) and f(K+i)
    and finally into a single utility function f(Ki), which balances an
    alternative's closeness to both the ideal and the anti-ideal. Higher
    scores indicate better alternatives.

    Parameters
    ----------
    defuzzify : callable, default=graded_mean_average_defuzzification
        Function used to defuzzify the TFN utility degrees K-i, K+i into
        crisp values.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Stanković, M., Stević, Ž., Das, D. K., Subotić, M., &
    Pamučar, D. (2020). A new fuzzy MARCOS method for road traffic risk
    analysis. Mathematics, 8(3), 457.
    """

    def __init__(
        self, 
        defuzzify: callable = graded_mean_average_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy MARCOS method object.

        Parameters
        ----------
        defuzzify : callable, default=graded_mean_average_defuzzification
            Function used to defuzzify Triangular Fuzzy Numbers.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """
        super().__init__(logger=logger)
        self.defuzzify = defuzzify

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy MARCOS method.

        This override exists to give `fMARCOS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fMARCOS._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.

        Returns
        -------
        np.ndarray, shape (m,)
            Fuzzy MARCOS utility score f(Ki) for each alternative; higher
            is better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent.
        RuntimeError
            If any computation step fails unexpectedly.

        Examples
        --------
        >>> matrix = np.array([
        ...     [[3, 4, 5], [4, 5, 6], [8, 9, 9]],
        ...     [[6, 7, 8], [4, 5, 6], [2, 3, 4]],
        ...     [[3, 4, 5], [5, 6, 7], [6, 7, 8]],
        ...     [[7, 8, 9], [6, 7, 8], [4, 5, 6]],
        ... ])
        >>> weights = np.array([[5, 7, 9], [7, 9, 9], [3, 5, 7]])
        >>> types = np.array([1, 1, 1])
        >>> fmarcos = fMARCOS()
        >>> scores = fmarcos(matrix, weights, types)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            *args,
            **kwargs
        )

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        MARCOS method.

        The decision matrix is extended with fuzzy Anti-Ideal (AAI) and
        Ideal (AI) rows, normalized, and weighted. Utility degrees relative
        to the anti-ideal (K-) and the ideal (K+) are computed for every
        alternative, defuzzified, and combined into per-degree utility
        functions f(K-), f(K+) and finally into the overall MARCOS utility
        function f(Ki).

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape ``(n,)``) or fuzzy
            (shape ``(n, 3)``). Crisp weights are automatically converted
            to triangular fuzzy weights.
        types : np.ndarray, shape (n,)
            Criteria types (``1`` for benefit criteria, ``-1`` for cost
            criteria).

        Returns
        -------
        np.ndarray, shape (m,)
            Final MARCOS utility score f(Ki) for each alternative. Higher
            values indicate better alternatives.

        Raises
        ------
        RuntimeError
            If extending the matrix, normalization, weighting, utility
            degree computation, defuzzification, or aggregation fails
            unexpectedly.
        """
        m, n, _ = matrix.shape

        # ── Step 1: AAI and AI solutions ─────────────────────────────
        try:
            aai = np.zeros((1, n, 3))
            ai = np.zeros((1, n, 3))

            for j in range(n):
                col = matrix[:, j, :]

                if types[j] == 1:      # benefit
                    aai[0, j] = min(col, key=lambda x: x[0])
                    ai[0, j] = max(col, key=lambda x: x[2])
                else:                  # cost
                    aai[0, j] = max(col, key=lambda x: x[2])
                    ai[0, j] = min(col, key=lambda x: x[0])

            extended = np.concatenate([aai, matrix, ai], axis=0)

        except Exception as e:
            raise RuntimeError(f"Failed to create reference solutions: {e}") from e

        self._log_step(
            self.logger,
            'AAI and AI solutions',
            np.concatenate([aai, ai], axis=0),
            'Anti-ideal and ideal fuzzy reference solutions'
        )

        # ── Step 2-3: Normalization and weighting ────────────────────
        try:
            nmatrix = extended.copy()

            for i in range(extended.shape[0]):
                for j in range(n):
                    if types[j] == 1:
                        nmatrix[i, j] = extended[i, j] / ai[0, j, 2]
                    else:
                        nmatrix[i, j] = aai[0, j, 0] / extended[i, j, ::-1]

            wmatrix = nmatrix * weights

        except Exception as e:
            raise RuntimeError(f"Failed during normalization and weighting: {e}") from e
            
        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Normalized extended matrix')
        self._log_step(self.logger, 'Weighted matrix', wmatrix, 'Weighted normalized matrix')

        # ── Step 4: Utility sums ─────────────────────────────────────
        try:
            S = np.sum(wmatrix, axis=1)

            S_aai = S[0]
            S_ai = S[-1]
            S_alt = S[1:-1]

        except Exception as e:
            raise RuntimeError(f"Failed to compute utility sums: {e}") from e

        # ── Step 5: Utility ratios ───────────────────────────────────
        try:
            K_minus = S_alt / S_aai[::-1]
            K_plus = S_alt / S_ai[::-1]

        except Exception as e:
            raise RuntimeError(f"Failed to compute utility ratios: {e}") from e

        self._log_step(self.logger, 'K-', K_minus, 'Utility relative to anti-ideal')
        self._log_step(self.logger, 'K+', K_plus, 'Utility relative to ideal')

        # ── Step 6: Final utility ────────────────────────────────────
        try:
            total = K_minus + K_plus

            D = np.max(total, axis=0)
            D_crisp = self.defuzzify(D)

            fK_plus = K_minus / D_crisp
            fK_minus = K_plus / D_crisp

            cK_plus = np.array([self.defuzzify(kp) for kp in K_plus])
            cfK_plus = np.array([self.defuzzify(fkp) for fkp in fK_plus])

            cK_minus = np.array([self.defuzzify(km) for km in K_minus])
            cfK_minus = np.array([self.defuzzify(fkm) for fkm in fK_minus])

        except Exception as e:
            raise RuntimeError(f"Failed to compute intermediate utility values: {e}") from e

        self._log_step(
            self.logger,
            'K- + K+',
            total,
            'Combined utility ratios relative to anti-ideal and ideal solutions'
        )

        self._log_step(
            self.logger,
            'D',
            D,
            'Maximum fuzzy utility values used for normalization'
        )

        self._log_step(
            self.logger,
            'Defuzzified D',
            np.array([D_crisp]),
            'Crisp normalization constant obtained from D'
        )

        self._log_step(
            self.logger,
            'f(K+)',
            fK_plus,
            'Normalized utility function relative to the anti-ideal solution'
        )

        self._log_step(
            self.logger,
            'f(K-)',
            fK_minus,
            'Normalized utility function relative to the ideal solution'
        )

        self._log_step(
            self.logger,
            'Defuzzified K+',
            cK_plus,
            'Defuzzified utility ratios relative to the ideal solution'
        )

        self._log_step(
            self.logger,
            'Defuzzified K-',
            cK_minus,
            'Defuzzified utility ratios relative to the anti-ideal solution'
        )

        self._log_step(
            self.logger,
            'Defuzzified f(K+)',
            cfK_plus,
            'Defuzzified normalized utility function f(K+)'
        )

        self._log_step(
            self.logger,
            'Defuzzified f(K-)',
            cfK_minus,
            'Defuzzified normalized utility function f(K-)'
        )

        try:
            utility = (cK_plus + cK_minus) / (1 + (1 - cfK_plus) / cfK_plus + (1 - cfK_minus) / cfK_minus)

        except Exception as e:
            raise RuntimeError(f"Failed to compute final utility: {e}") from e

        self._log_step(
            self.logger,
            'Utility function f(Ki)',
            utility,
            'Final MARCOS utility scores; higher values indicate better alternatives'
        )

        return utility