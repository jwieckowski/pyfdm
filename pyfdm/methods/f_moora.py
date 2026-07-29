# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import vector_normalization


class fMOORA(BaseFuzzyMethod):
    """
    Fuzzy Multi-Objective Optimization on the basis of Ratio Analysis
    (MOORA) — ratio system approach.

    The decision matrix is normalized (vector normalization is the
    standard choice for MOORA) and weighted. For each alternative, the
    weighted normalized values of benefit criteria are summed (Sp) and
    the weighted normalized values of cost criteria are summed separately
    (Sm). The final ratio score is the (fuzzy) difference Sp - Sm,
    defuzzified into a crisp preference value. Higher scores indicate
    better alternatives.

    Parameters
    ----------
    normalization : callable, default=vector_normalization
        Function used to normalize the TFN decision matrix. Vector
        normalization is the normalization originally associated with
        MOORA's ratio system.
    logger : StepLogger | None, optional
        Optional logger used for recording intermediate computation steps.

    References
    ----------
    Karande, P., & Chakraborty, S. (2012). A Fuzzy-MOORA 
    approach for ERP system selection. Decision Science 
    Letters, 1(1), 11-21.
    """

    def __init__(
        self, 
        normalization: callable=vector_normalization, 
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy MOORA method object.

        Parameters
        ----------
        normalization : callable, default=vector_normalization
            Function used to normalize the TFN decision matrix.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.

        """
        super().__init__(logger=logger)
        self.normalization = normalization

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy MOORA method.

        This override exists to give `fMOORA` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fMOORA._calculate`.

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
            Fuzzy MOORA preference score for each alternative; higher is
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
        >>> fmoora = fMOORA()
        >>> scores = fmoora(matrix, weights, types)
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
        MOORA (ratio system) method.

        The decision matrix is normalized and weighted; weighted values
        are then split and summed separately for benefit criteria (Sp) and
        cost criteria (Sm), and combined into a final ratio score.

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
            Ratio score for each alternative. Higher values indicate
            better alternatives.

        Raises
        ------
        RuntimeError
            If normalization, weighting, the benefit/cost sums, or the
            final score computation fails unexpectedly.
        """
        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Result of applying normalization (vector normalization recommended for MOORA)')

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wmatrix = nmatrix * weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights to the normalized matrix: {e}") from e

        self._log_step(self.logger, 'Weighted normalized matrix', wmatrix, 'Element-wise product of normalized matrix and weights')

        try:
            Sp = np.sum(wmatrix[:, types == 1], axis=1)
            Sm = np.sum(wmatrix[:, types == -1], axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to compute benefit/cost sums (Sp, Sm): {e}") from e

        self._log_step(self.logger, 'Profit sum (Sp)', np.array(Sp, dtype=float), 'Sum of weighted normalized values for profit criteria')
        self._log_step(self.logger, 'Cost sum (Sm)', np.array(Sm, dtype=float), 'Sum of weighted normalized values for cost criteria')

        try:
            S = np.array([np.sqrt(1/3 * ((sp[0]-sm[0])**2 + (sp[1]-sm[1])**2 + (sp[2]-sm[2])**2))
                          for sp, sm in zip(Sp, Sm)])
        except Exception as e:
            raise RuntimeError(f"Failed to compute preference scores: {e}") from e

        self._log_step(self.logger, 'Preference scores (S)', S, 'Combined Sp/Sm score — higher is better (see class Notes on the aggregation formula)')
        return S
