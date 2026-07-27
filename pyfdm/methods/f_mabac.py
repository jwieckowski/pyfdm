# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import mean_defuzzification
from ..utils import minmax_normalization


class fMABAC(BaseFuzzyMethod):
    """
    Fuzzy Multi-Attributive Border Approximation Area Comparison (MABAC)
    method.

    The method evaluates alternatives by measuring their distances from a
    fuzzy Border Approximation Area (BAA). The decision matrix is first
    normalized and weighted, after which the BAA is determined as the
    geometric mean of weighted criterion values across all alternatives.
    Each alternative is then compared against the BAA, producing a distance
    matrix whose aggregated values are finally defuzzified into crisp
    preference scores. Higher scores indicate better alternatives.

    .. rubric:: Reference

        Zolfani, S. H., Görçün, Ö. F., & Küçükönder, H. (2021). Evaluating
        logistics villages in Turkey using hybrid improved fuzzy SWARA (IMF SWARA)
        and fuzzy MABAC techniques. Technological and Economic Development of
        Economy, 27(6), 1582-1612.


    Parameters
    ----------
    normalization : callable, default=minmax_normalization
        Function used to normalize the TFN decision matrix.
    defuzzify : callable, default=mean_defuzzification
        Function used to transform the final fuzzy preference values into
        crisp scores.
    logger : StepLogger | None, optional
        Optional logger used for recording intermediate computation steps.

    """

    def __init__(
        self, 
        normalization: callable=minmax_normalization, 
        defuzzify: callable=mean_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy MABAC method object.

        Parameters
        ----------
        normalization : callable, default=minmax_normalization
            Function used to normalize the TFN decision matrix.
        defuzzify : callable, default=mean_defuzzification
            Function used to transform the final fuzzy preference values into
            crisp scores.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.

        """
        super().__init__(logger=logger)
        self.normalization = normalization
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
        Execute the fuzzy MABAC method.

        This override exists to give `fMABAC` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fMABAC._calculate`.

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
            fuzzy MABAC preference score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent.
        RuntimeError
            If any computation step fails unexpectedly.

        Examples
        --------
        >>> matrix = np.array([
            [[3, 4, 5], [4, 5, 6], [8, 9, 9]],
            [[6, 7, 8], [4, 5, 6], [2, 3, 4]],
            [[3, 4, 5], [5, 6, 7], [6, 7, 8]],
            [[7, 8, 9], [6, 7, 8], [4, 5, 6]],
        ])
        >>> weights = np.array([[5, 7, 9], [7, 9, 9], [3, 5, 7]])
        >>> types = np.array([1, 1, 1])
        >>> fmabac = fMABAC()
        >>> scores = fmabac(matrix, weights, types)
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
        MABAC (Multi-Attributive Border Approximation Area Comparison)
        method.

        The decision matrix is first normalized and weighted according to
        the MABAC weighting scheme. A fuzzy Border Approximation Area (BAA)
        is then determined for each criterion using the geometric mean of
        weighted values. Finally, each alternative's distance from the BAA
        is aggregated and defuzzified to obtain a crisp preference score.

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
            criteria), forwarded to the selected normalization function.

        Returns
        -------
        np.ndarray, shape (m,)
            Crisp preference score for each alternative. Higher values
            indicate better alternatives.

        Raises
        ------
        RuntimeError
            If normalization, weighting, border approximation area
            computation, distance calculation, aggregation, or
            defuzzification fails unexpectedly.
        """
        
        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(
            self.logger,
            'Normalized matrix',
            nmatrix,
            'Result of applying normalization function'
        )

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wmatrix = nmatrix * weights + weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply the MABAC weighting scheme: {e}") from e

        self._log_step(
            self.logger,
            'Weighted normalized matrix',
            wmatrix,
            'nmatrix * weights + weights (MABAC-specific weighting)'
        )

        try:
            G = np.prod(wmatrix, axis=0) ** (1 / wmatrix.shape[0])
        except Exception as e:
            raise RuntimeError(f"Failed to compute the Border Approximation Area (G): {e}") from e

        self._log_step(
            self.logger,
            'Border Approximation Area (G)',
            np.array(G, dtype=float),
            'Geometric mean across alternatives per criterion'
        )

        try:
            Q = wmatrix - G[..., ::-1]
        except Exception as e:
            raise RuntimeError(f"Failed to compute distances from the Border Approximation Area: {e}") from e

        self._log_step(
            self.logger,
            'Distance matrix (Q)',
            np.array(Q, dtype=float),
            'Distance of each alternative from the border approximation area'
        )

        try:
            S = np.array([np.sum(q, axis=0) for q in Q])
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate distances for each alternative: {e}") from e

        self._log_step(
            self.logger,
            'Sum of distances (S)',
            np.array(S, dtype=float),
            'Row sums of the distance matrix'
        )

        try:
            result = np.array([self.defuzzify(s) for s in S])
        except Exception as e:
            raise RuntimeError(f"Failed to defuzzify preference scores: {e}") from e

        self._log_step(
            self.logger,
            'Preference scores (defuzzified)',
            result,
            'Defuzzified S values — higher is better'
        )

        return result
