# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import linear_normalization, vertex_distance

class fTOPSIS(BaseFuzzyMethod):
    """
    Fuzzy Technique for Order Preference by Similarity to Ideal Solution
    (fuzzy TOPSIS).

    The method normalizes the fuzzy decision matrix, applies fuzzy criteria
    weights, determines the distances of alternatives from the Fuzzy Positive
    Ideal Solution (FPIS) and Fuzzy Negative Ideal Solution (FNIS), and
    calculates the closeness coefficient:

    ``CC_i = D_i^- / (D_i^+ + D_i^-)``

    Higher values indicate better alternatives.

    .. rubric:: Reference
    
        Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making
        under fuzzy environment. Fuzzy sets and systems, 114(1), 1-9.

    Parameters
    ----------
    normalization : callable, default=linear_normalization
        Function used for normalization of the fuzzy decision matrix.

    distance : callable, default=vertex_distance
        Distance function used to calculate the distance between TFNs.

    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """

    def __init__(
        self,
        normalization=linear_normalization,
        distance=vertex_distance,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy TOPSIS method object.

        Parameters
        ----------
        normalization : callable, default=linear_normalization
            Function used for fuzzy matrix normalization.

        distance : callable, default=vertex_distance
            Function used to calculate distances between fuzzy numbers.

        logger : StepLogger | None, optional
            Optional logger used for storing computation steps.
        """

        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance = distance

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute fuzzy TOPSIS.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            Fuzzy decision matrix containing TFNs.

        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criteria weights. Crisp weights are automatically transformed
            into singleton fuzzy numbers.

        types : np.ndarray | list, shape (n,)
            Criteria types:
            ``1`` - benefit criterion,
            ``-1`` - cost criterion.

        Returns
        -------
        np.ndarray, shape (m,)
            Closeness coefficients of alternatives.
            Higher values indicate better alternatives.

        Raises
        ------
        ValueError
            If input data are invalid.

        RuntimeError
            If a computational step fails.
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
        types: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate fuzzy TOPSIS preference scores.

        Parameters
        ----------
        matrix : np.ndarray
            TFN decision matrix.

        weights : np.ndarray
            Crisp or fuzzy criteria weights.

        types : np.ndarray
            Criteria types.

        Returns
        -------
        np.ndarray
            Closeness coefficients.

        Raises
        ------
        RuntimeError
            If any computational stage fails.
        """

        # -----------------------------
        # Normalization
        # -----------------------------
        try:
            nmatrix = self.normalization(matrix, types)

        except Exception as e:
            raise RuntimeError(f"Fuzzy normalization failed: {e}") from e

        self._log_step(
            self.logger,
            'Normalized fuzzy decision matrix',
            np.array(nmatrix, dtype=float),
            'Result of applying the normalization function'
        )

        # Weighting
        try:
            if weights.ndim == 1:
                weights = np.repeat(weights, 3).reshape((len(weights), 3))
            wmatrix = nmatrix * weights

        except Exception as e:
            raise RuntimeError(f"Failed to apply criteria weights: {e}") from e

        self._log_step(
            self.logger,
            'Weighted normalized matrix',
            np.array(wmatrix, dtype=float),
            'Element-wise multiplication of normalized values and fuzzy weights'
        )

        # Ideal solutions
        try:
            ideal = np.ones((matrix.shape[1], matrix.shape[2]))
            nideal = np.zeros((matrix.shape[1], matrix.shape[2]))

        except Exception as e:
            raise RuntimeError(f"Failed to create fuzzy ideal solutions: {e}") from e

        self._log_step(
            self.logger,
            'Fuzzy Positive Ideal Solution (FPIS)',
            ideal,
            'Upper reference point consisting of ones'
        )

        self._log_step(
            self.logger,
            'Fuzzy Negative Ideal Solution (FNIS)',
            nideal,
            'Lower reference point consisting of zeros'
        )


        # Distances
        try:
            d_positive = np.zeros(matrix.shape[0])
            d_negative = np.zeros(matrix.shape[0])

            for i in range(matrix.shape[0]):
                d_positive[i] = np.sum([
                    self.distance(
                        wmatrix[i, j],
                        ideal[j]
                    )
                    for j in range(matrix.shape[1])
                ])

                d_negative[i] = np.sum([
                    self.distance(
                        wmatrix[i, j],
                        nideal[j]
                    )
                    for j in range(matrix.shape[1])
                ])

        except Exception as e:
            raise RuntimeError(f"Failed to calculate distances from ideal solutions: {e}") from e

        self._log_step(
            self.logger,
            'Distance to FPIS',
            np.array(d_positive, dtype=float),
            'Distance of alternatives from fuzzy positive ideal solution'
        )

        self._log_step(
            self.logger,
            'Distance to FNIS',
            np.array(d_negative, dtype=float),
            'Distance of alternatives from fuzzy negative ideal solution'
        )

        # Closeness coefficient
        try:
            result = d_negative / (d_positive + d_negative)
        except Exception as e:
            raise RuntimeError(f"Failed to compute TOPSIS closeness coefficient: {e}") from e

        self._log_step(
            self.logger,
            'Preference scores (CC)',
            np.array(result, dtype=float),
            'Closeness coefficient CC = D- / (D+ + D-)'
        )

        return result
