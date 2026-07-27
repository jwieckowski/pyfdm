# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import vector_normalization, graded_mean_average_defuzzification

class fPIV(BaseFuzzyMethod):
    """
    Fuzzy PIV — Proximity Indexed Value.

    The decision matrix is vector-normalized and weighted. For each
    criterion, a "best value" (bfv) is determined -- the componentwise
    maximum across alternatives for benefit criteria, the componentwise
    minimum for cost criteria. Each alternative's proximity to that best
    value is then computed per criterion (the gap `bfv - v_ij` for
    benefit criteria, `v_ij - bfv` for cost criteria), summed across
    criteria, and defuzzified into a single overall proximity
    index. A LOWER proximity index means the alternative is closer to
    the positive ideal solution, so ranking is ascending.

    .. rubric:: Reference
        Seraj, M., Yahya, S. M., Badruddin, I. A., Anqi, A. E., Asjad, 
        M., & Khan, Z. A. (2019). Multi-response optimization of 
        nanofluid-based IC engine cooling system using fuzzy 
        method. Processes, 8(1), 30.

    Parameters
    ----------
    normalization : callable, default=vector_normalization
        Function used to normalize the TFN decision matrix. Called as
        ``normalization(matrix, types)``.
    defuzzify : callable, default=graded_mean_average_defuzzification
        Function used to defuzzify the final aggregate TFN performance
        score into a crisp value.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """

    _descending = False 

    def __init__(
        self, 
        normalization: callable = vector_normalization, 
        defuzzify: callable = graded_mean_average_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy PIV method object.

        Parameters
        ----------
        normalization : callable, default=vector_normalization
            Function used to normalize the TFN decision matrix.
        defuzzify : callable, default=graded_mean_average_defuzzification
            Function used to defuzzify Triangular Fuzzy Numbers.
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
        Execute the fuzzy PIV method.

        This override exists to give `fPIV` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fPIV._calculate`.

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
            Fuzzy PIV proximity index for each alternative; LOWER is
            better (see `_descending`).

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
        >>> fpiv = fPIV()
        >>> scores = fpiv(matrix, weights, types)
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
        PIV method.

        The decision matrix is normalized and weighted; a per-criterion
        best value (componentwise max for benefit criteria, min for cost
        criteria) is derived, and each alternative's proximity to that
        best value is summed across criteria and defuzzified.

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
            Proximity index for each alternative. LOWER values indicate
            better alternatives (closer to the ideal).

        Raises
        ------
        RuntimeError
            If normalization, weighting, the best-value computation, the
            proximity calculation, or defuzzification fails unexpectedly.
        """

        m, n, _ = matrix.shape

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Result of applying normalization function')

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wnmatrix = nmatrix * weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights to the normalized matrix: {e}") from e

        self._log_step(self.logger, 'Weighted normalized matrix', wnmatrix, 'Element-wise product of normalized matrix and weights')

        try:
            bfv = np.zeros((n, 3))
            for j in range(n):
                if types[j] == 1:
                    bfv[j] = np.max(wnmatrix[:, j], axis=0)
                else:
                    bfv[j] = np.min(wnmatrix[:, j], axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute the best values (bfv): {e}") from e

        self._log_step(self.logger, 'Best values (bfv)', bfv, 'Componentwise max (benefit) / min (cost) across alternatives per criterion')

        try:
            fpi = np.zeros((m, n, 3))
            for i in range(m):
                for j in range(n):
                    if types[j] == 1:
                        fpi[i, j] = bfv[j] - wnmatrix[i, j]
                    else: 
                        fpi[i, j] = wnmatrix[i, j] - bfv[j]
        except Exception as e:
            raise RuntimeError(f"Failed to compute the proximity matrix (fpi): {e}") from e

        self._log_step(self.logger, 'Proximity matrix (fpi)', fpi, 'Gap between each alternative and the best value per criterion')

        try:
            d = np.sum(fpi, axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate the proximity index (d): {e}") from e

        self._log_step(self.logger, 'Aggregated proximity index (d)', d, 'Row sums of the proximity matrix')

        try:
            opi = [self.defuzzify(dd) for dd in d]
        except Exception as e:
            raise RuntimeError(f"Defuzzification failed: {e}") from e

        self._log_step(self.logger, 'Overall proximity index (opi)', opi, 'Defuzzified proximity index — lower is better')
        return opi