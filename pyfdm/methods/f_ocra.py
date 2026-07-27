# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import mean_defuzzification

class fOCRA(BaseFuzzyMethod):
    """
    Fuzzy OCRA — Operational Competitiveness Rating.

    Alternatives are rated separately on cost criteria and benefit
    (profit) criteria, each relative to the best-performing (minimum cost
    / maximum benefit-consistent) reference derived from the decision
    matrix. For cost criteria, a preference rating (Is) measures how much
    better an alternative is than the worst-cost alternative, scaled by
    the best (minimum) cost; for benefit criteria, an analogous rating
    (Os) measures the improvement over the worst-benefit alternative,
    scaled the same way. Both ratings are linearized (shifted so the
    worst alternative scores zero) and combined into an overall
    performance score, again shifted to a zero baseline and defuzzified.
    Higher scores indicate better alternatives.

    .. rubric:: Reference
    
        ULUTAŞ, A. (2019). Supplier selection by using a fuzzy integrated 
        model for a textile company. Engineering Economics, 30(5), 579-590.

    Parameters
    ----------
    defuzzify : callable, default=mean_defuzzification
        Function used to defuzzify the final aggregate TFN performance
        score into a crisp value.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """


    def __init__(
        self, 
        defuzzify: callable = mean_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy OCRA method object.

        Parameters
        ----------
        defuzzify : callable, default=mean_defuzzification
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
        Execute the fuzzy OCRA method.

        This override exists to give `fOCRA` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fOCRA._calculate`.

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
            Fuzzy OCRA preference score for each alternative; higher is
            better.

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
        >>> focra = fOCRA()
        >>> scores = focra(matrix, weights, types)
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
        OCRA method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape ``(n,)``) or fuzzy
            (shape ``(n, 3)``); both broadcast correctly against
            per-criterion TFN terms.
        types : np.ndarray, shape (n,)
            Criteria types (``1`` for benefit criteria, ``-1`` for cost
            criteria).

        Returns
        -------
        np.ndarray, shape (m,)
            Final OCRA performance score for each alternative. Higher
            values indicate better alternatives.

        Raises
        ------
        RuntimeError
            If computing the cost/benefit ratings, linearization,
            aggregation, or defuzzification fails unexpectedly.
        """

        try:
            Is = np.zeros((matrix.shape[0], matrix.shape[2]))
            for i in range(matrix.shape[0]):
                Is[i] = np.sum([
                    weights[j] * (
                        (np.max(matrix[:, j], axis=0) - matrix[i, j][..., ::-1]) /
                        np.min(matrix[:, j], axis=0)
                    )
                    for j in range(matrix.shape[1])
                    if types[j] == -1
                ], axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute cost performance ratings (Is): {e}") from e

        self._log_step(
            self.logger,
            'Cost performance rating (Is)',
            Is,
            'Scaled cost improvements relative to the minimum cost alternative'
        )

        try:
            Iss = Is - np.min(Is, axis=0)[..., ::-1]
        except Exception as e:
            raise RuntimeError(f"Failed to linearize cost performance ratings (Iss): {e}") from e

        self._log_step(
            self.logger,
            'Linear cost rating (Iss)',
            Iss,
            'Linearized cost rating: Is minus column minimum'
        )

        try:
            Os = np.zeros((matrix.shape[0], matrix.shape[2]))
            for i in range(matrix.shape[0]):
                Os[i] = np.sum([
                    weights[j] * (
                        (matrix[i, j] - np.min(matrix[:, j][..., ::-1], axis=0)) /
                        np.min(matrix[:, j], axis=0)
                    )
                    for j in range(matrix.shape[1])
                    if types[j] == 1
                ], axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute benefit performance ratings (Os): {e}") from e

        self._log_step(
            self.logger,
            'Profit performance rating (Os)',
            Os,
            'Scaled profit improvements relative to the minimum profit alternative'
        )

        try:
            Oss = Os - np.min(Os, axis=0)[..., ::-1]
        except Exception as e:
            raise RuntimeError(f"Failed to linearize benefit performance ratings (Oss): {e}") from e

        self._log_step(
            self.logger,
            'Linear profit rating (Oss)',
            Oss,
            'Linearized profit rating: Os minus column minimum'
        )

        try:
            P = Iss + Oss - np.min(Iss + Oss, axis=0)[..., ::-1]
        except Exception as e:
            raise RuntimeError(f"Failed to compute aggregate performance ratings (P): {e}") from e

        self._log_step(
            self.logger,
            'Aggregate performance (P)',
            P,
            'Combined cost and profit ratings, shifted to zero base'
        )

        try:
            result = np.array([self.defuzzify(p) for p in P])
        except Exception as e:
            raise RuntimeError(f"Failed to defuzzify preference scores: {e}") from e

        self._log_step(
            self.logger,
            'Preference scores (defuzzified)',
            result,
            'Defuzzified aggregate performance — higher is better'
        )

        return result
