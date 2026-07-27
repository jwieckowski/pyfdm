# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import saw_normalization


class fCOPRAS(BaseFuzzyMethod):
    """
    Fuzzy COmplex PRoportional ASsessment (COPRAS) method.

    The method normalizes the fuzzy decision matrix, applies criterion
    weights, separately aggregates benefit and cost criteria, and computes
    the relative significance of each alternative. The fuzzy significance
    values are then defuzzified and normalized to obtain the final
    preference scores. Higher scores indicate better alternatives.

    .. rubric:: Reference

        Narang, M., Joshi, M. C., & Pal, A. K. (2021). A hybrid fuzzy 
        COPRAS-base-criterion method for multi-criteria decision making.
        Soft Computing, 25(13), 8391-8399.

    Parameters
    ----------
    normalization : callable
        Function used to normalize the fuzzy decision matrix.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """

    _different_types_required = True

    def __init__(
        self,
        normalization=saw_normalization,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy COPRAS method object.

        Parameters
        ----------
        normalization : callable, default=saw_normalization
            Function used to normalize the fuzzy decision matrix.
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
        Execute the fuzzy COPRAS method.

        This override exists to give `fCOPRAS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fCOPRAS._calculate`.

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
            fuzzy COPRAS preference score for each alternative; higher is better.

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
        >>> types = np.array([1, -1, 1])
        >>> fcopras = fCOPRAS()
        >>> scores = fcopras(matrix, weights, types)
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
        COPRAS method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp or fuzzy.
        types : np.ndarray, shape (n,)
            Criteria types (1 for benefit, -1 for cost).

        Returns
        -------
        np.ndarray, shape (m,)
            Preference score for each alternative. Higher values indicate
            better alternatives.

        Raises
        ------
        RuntimeError
            If any computation stage fails.
        """

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(
                f"Normalization step failed: {e}"
            ) from e

        self._log_step(
            self.logger,
            "Normalized matrix",
            nmatrix,
            "Result of applying the normalization function."
        )

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wmatrix = nmatrix * weights
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply weights to the normalized matrix: {e}"
            ) from e

        self._log_step(
            self.logger,
            "Weighted normalized matrix",
            wmatrix,
            "Element-wise product of the normalized matrix and criterion weights."
        )

        try:
            Tp = np.sum(wmatrix[:, types == 1], axis=1)
            Tm = np.sum(wmatrix[:, types == -1], axis=1)
        except Exception as e:
            raise RuntimeError(
                f"Failed to aggregate benefit and cost criteria: {e}"
            ) from e

        self._log_step(
            self.logger,
            "Benefit aggregation (Tp)",
            np.asarray(Tp, dtype=float),
            "Sum of weighted normalized benefit criteria."
        )

        self._log_step(
            self.logger,
            "Cost aggregation (Tm)",
            np.asarray(Tm, dtype=float),
            "Sum of weighted normalized cost criteria."
        )

        try:
            Q = Tp + (np.sum(Tm)) / (Tm * np.sum(np.divide(1,Tm.astype(float))))
            if np.isnan(np.max(Q.astype(float).ravel())):
                Q = np.nan_to_num(Q.ravel().astype(float)).reshape(Tp.shape)

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute relative significance values: {e}"
            ) from e

        self._log_step(
            self.logger,
            "Relative significance (Q)",
            np.asarray(Q, dtype=float),
            "Combined benefit and cost utility before defuzzification."
        )

        try:
            Q = Q[:, 0] + ((Q[:, 2] - Q[:, 0]) - (Q[:, 1] - Q[:, 2])) / 3
            preferences = Q / np.max(Q) 

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute final preference scores: {e}"
            ) from e

        self._log_step(
            self.logger,
            "Preference scores",
            preferences,
            "Defuzzified and normalized COPRAS preference values. Higher values indicate better alternatives."
        )

        return preferences