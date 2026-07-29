# Copyright (c) 2023-2026 Jakub Więckowski, Andrii Shekhovtsov

import numpy as np
from typing import Any
from functools import reduce

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..validator import Validator
from ..TFN import TFN

class fSPOTIS(BaseFuzzyMethod):
    """
    Fuzzy Stable Preference Ordering Towards Ideal Solution (fSPOTIS).

    The method evaluates alternatives by calculating their normalized
    distance from the Ideal Solution Point (ISP). The distances are
    calculated using fuzzy numbers and aggregated using weighted fuzzy
    membership aggregation.

    Lower preference values indicate alternatives closer to the ISP,
    therefore the ranking direction is descending internally reversed
    according to the SPOTIS principle.

    Parameters
    ----------
    normalization : callable | None, optional
        Function used for normalization of the fuzzy decision matrix.
        If None, the original matrix is used.

    bounds : np.ndarray, optional, default = None
        Criteria bounds defining the space of the decision problem for evaluation.
        If None, they are calculated automatically.

    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Shekhovtsov, A., Paradowski, B., Więckowski, J., Kizielewicz, B.,
    & Sałabun, W. (2022, December). Extension of the SPOTIS method
    for the rank reversal free decision-making under fuzzy 
    environment. In 2022 IEEE 61st Conference on Decision 
    and Control (CDC) (pp. 5595-5600). IEEE.
    """

    _crisp_weights_required = True

    def __init__(
        self,
        normalization: callable = None,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy SPOTIS method object.

        Parameters
        ----------
        normalization : callable, optional, default=None
            Function used to calculate the normalized decision matrix.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """

        super().__init__(logger=logger)
        self.normalization = normalization

    def make_bounds(
        self,
        matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate criterion bounds from fuzzy decision matrix.

        Parameters
        ----------
        matrix : np.ndarray, shape (m,n,3)
            TFN decision matrix.

        Returns
        -------
        np.ndarray, shape (n,2)
            Lower and upper bounds for each criterion.
        """

        try:
            bounds = np.hstack((
                np.min(matrix[:, :, 0], axis=0).reshape(-1, 1),
                np.max(matrix[:, :, 2], axis=0).reshape(-1, 1),
            ))

            return bounds
        except Exception as e:
            raise RuntimeError(f"Failed to calculate SPOTIS bounds: {e}") from e

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        bounds: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy SPOTIS method.

        This override exists to give `fSPOTIS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fSPOTIS._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m,n,3)
            TFN decision matrix.

        weights : np.ndarray | list, shape (n,) or (n,3)
            Criterion weights.

        types : np.ndarray | list, shape (n,)
            Criterion types:
            ``1`` benefit criterion,
            ``-1`` cost criterion.

        bounds : np.ndarray, optional, default = None
            Criteria bounds defining the space of the decision problem for evaluation.
            If None, they are calculated automatically.

        Returns
        -------
        np.ndarray
            SPOTIS preference values.
            Lower values represent better alternatives.

        Raises
        ------
        ValueError
            If input validation fails.

        RuntimeError
            If a computational step fails.

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
        >>> f_spotis = fSPOTIS()
        >>> scores = f_spotis(matrix, weights, types)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            bounds=bounds,
            *args,
            **kwargs
        )

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        bounds: np.ndarray | None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy ERVD inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        bounds : np.ndarray, optional, default = None
            Criteria bounds defining the space of the decision problem for evaluation.
            If None, they are calculated automatically.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If input dimensions are inconsistent.
        TypeError
            If data cannot be converted to numeric arrays.
        """

        try:
            super()._validate_input(
                matrix,
                weights,
                types,
                self._crisp_weights_required,
                self._different_types_required
            )
            Validator.validate_spotis_input(matrix, bounds)

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fSPOTIS input: {e}") from e

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        bounds: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate fuzzy SPOTIS preference values.

        Parameters
        ----------
        matrix : np.ndarray
            TFN decision matrix.

        weights : np.ndarray
            Criterion weights.

        types : np.ndarray
            Criterion types.

        bounds : np.ndarray, optional
            Criterion bounds. If None, they are calculated automatically.

        Returns
        -------
        np.ndarray
            SPOTIS preference values.
            Lower values indicate better alternatives.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If the normalization step or the utility/score computation
            fails unexpectedly.
        """
        # Bounds
        try:
            if bounds is None:
                bounds = self.make_bounds(matrix)

        except Exception as e:
            raise RuntimeError(f"Failed to determine bounds: {e}") from e

        self._log_step(
            self.logger,
            "Criterion bounds",
            bounds,
            "Minimum and maximum values defining the SPOTIS solution space"
        )

        # Normalization
        try:
            nmatrix = (
                self.normalization(matrix, types)
                if self.normalization
                else matrix.copy()
            )

        except Exception as e:
            raise RuntimeError(f"Normalization failed: {e}") from e

        self._log_step(
            self.logger,
            "Normalized matrix",
            nmatrix,
            "Normalized fuzzy decision matrix"
        )

        # Ideal Solution Point
        try:
            isp = bounds[np.arange(bounds.shape[0]), ((types + 1) // 2).astype(int)]

        except Exception as e:
            raise RuntimeError(f"Failed to calculate ISP: {e}") from e

        self._log_step(
            self.logger,
            "Ideal Solution Point (ISP)",
            isp,
            "Reference point used for distance calculation"
        )

        # Distance matrix
        try:

            tf_matrix = np.array([[TFN(*x) for x in row] for row in nmatrix])
            d_matrix = tf_matrix.copy()

            for j in range(len(isp)):
                d_matrix[:, j] = np.abs(
                    (tf_matrix[:, j] - isp[j]) /
                    (bounds[j, 1] - bounds[j, 0])
                )

        except Exception as e:
            raise RuntimeError(f"Distance calculation failed: {e}") from e

        self._log_step(
            self.logger,
            "Fuzzy distance matrix",
            d_matrix,
            "Normalized distances from Ideal Solution Point"
        )

        # Aggregation
        def algsum(a, b):
            return a + b - a * b

        def aggregate(row):
            x = np.linspace(np.min(row).a, np.max(row).c, 512)
            mu = [w * t.membership_function(x) for w, t in zip(weights, row)]
            agg = reduce(algsum, mu)
            return np.sum(x * agg) / np.sum(agg)

        try:
            result = np.array([aggregate(row) for row in d_matrix])

        except Exception as e:
            raise RuntimeError(f"Preference calculation failed: {e}") from e

        self._log_step(
            self.logger,
            "Preference scores",
            result,
            "Final fuzzy SPOTIS scores; lower values indicate better alternatives"
        )

        return result
