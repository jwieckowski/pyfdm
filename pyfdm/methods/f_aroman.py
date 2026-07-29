# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..validator import Validator

from ..utils.normalizations import minmax_normalization, vector_normalization
from ..utils import rank_alternatives, graded_mean_average_defuzzification

class fAROMAN(BaseFuzzyMethod):
    """
    Fuzzy Alternative Ranking Order Method Accounting for two-step
    Normalization (AROMAN).

    Combines a min-max normalization and a vector normalization of the TFN
    decision matrix into a single aggregated matrix (weighted by `beta`),
    applies criterion weights, and separately aggregates the weighted
    contributions of cost and benefit criteria (raised to complementary
    exponents derived from the weights) into a final preference score via
    ``R_i = e^(A_i - L_i)``. Higher scores indicate better alternatives.

    Parameters
    ----------
    beta : float, default=0.5
        Weight controlling the relative influence of the min-max
        normalization vs. the vector normalization in the aggregated
        matrix (``beta`` for min-max, ``1 - beta`` for vector).
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Čubranić-Dobrodolac, M., Jovčić, S., Bošković, S., & Babić, D. (2023).
    A decision-making model for professional drivers selection:
    A hybridized fuzzy-AROMAN-Fuller approach. Mathematics, 11(13), 2831.
    """

    def __init__(
        self,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy AROMAN method object.

        Parameters
        ----------
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """
        super().__init__(logger=logger)

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        beta: float = 0.5,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy AROMAN method.

        This override exists purely to surface AROMAN's specific tuning
        parameters (`beta`) in the call signature/docstring for IDEs and `help()`; 
        all actual validation, computation, and logging happen in 
        `BaseFuzzyMethod.__call__` / `fAROMAN._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        beta : float, default=0.5
            Weight used to blend the min-max and vector normalizations
            when building the aggregated matrix (see class Notes).

        Returns
        -------
        np.ndarray, shape (m,)
            AROMAN preference score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent, or `beta` fail validation (see `_calculate`).
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
        >>> faroman = fAROMAN()
        >>> scores = faroman(matrix, weights, types)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            beta=beta,
            *args,
            **kwargs
        )

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        beta: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy AROMAN inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        beta : float, default=0.5
            Weight used to blend the min-max and vector normalizations
            when building the aggregated matrix (see class Notes).

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
            Validator.validate_param_range(beta, 0, 1, 'beta')

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fAROMAN input: {e}") from e

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        beta: float = 0.5,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        AROMAN method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,), broadcast to TFN
            (w, w, w)) or already fuzzy (shape (n, 3)).
        types : np.ndarray, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        beta : float, default=0.5
            Weight used to blend the min-max and vector normalizations
            when building the aggregated matrix (see class Notes).

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy AROMAN preference scores ``R_i = e^(A_i - L_i)``; higher is
            better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If any computation step fails unexpectedly (e.g. due to
            non-finite intermediate values).

        """

        try:
            n1_matrix = minmax_normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Min-max normalization failed: {e}") from e

        self._log_step(
            self.logger,
            'Min-max normalized matrix',
            np.array(n1_matrix, dtype=float),
            'Result of min-max normalization'
        )

        try:
            n2_matrix = vector_normalization(matrix)
        except Exception as e:
            raise RuntimeError(f"Vector normalization failed: {e}") from e

        self._log_step(
            self.logger,
            'Vector normalized matrix',
            np.array(n2_matrix, dtype=float),
            'Result of vector normalization'
        )

        try:
            agg_matrix = (beta * n1_matrix + (1 - beta) * n2_matrix) / 2
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate normalized matrices: {e}") from e

        self._log_step(
            self.logger,
            'Aggregated normalized matrix',
            np.array(agg_matrix, dtype=float),
            f'Weighted aggregation of min-max and vector normalization (β = {beta})'
        )

        try:
            if weights.ndim == 1:
                weights = np.repeat(weights, 3).reshape((len(weights), 3))

            wagg_matrix = agg_matrix * weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights to the aggregated matrix: {e}") from e

        self._log_step(
            self.logger,
            'Weighted aggregated matrix',
            np.array(wagg_matrix, dtype=float),
            'Aggregated matrix multiplied by criteria weights'
        )

        try:
            lam = np.sum(np.min(weights, axis=0))
        except Exception as e:
            raise RuntimeError(f"Failed to compute lambda: {e}") from e

        self._log_step(
            self.logger,
            'Lambda (λ)',
            np.array([lam]),
            'Sum of lower bounds of fuzzy weights'
        )

        try:
            Li = np.sum(wagg_matrix[:, types == -1], axis=1) ** lam
        except Exception as e:
            raise RuntimeError(f"Failed to compute cost utility (L): {e}") from e

        self._log_step(
            self.logger,
            'Cost utility (L)',
            np.array(Li, dtype=float),
            'Aggregated contribution of cost criteria'
        )

        try:
            Ai = np.sum(wagg_matrix[:, types == 1], axis=1) ** (1 - lam)
        except Exception as e:
            raise RuntimeError(f"Failed to compute benefit utility (A): {e}") from e

        self._log_step(
            self.logger,
            'Benefit utility (A)',
            np.array(Ai, dtype=float),
            'Aggregated contribution of benefit criteria'
        )

        diff = Ai - Li
        self._log_step(
            self.logger,
            'Utility difference (A - L)',
            np.array(diff, dtype=float),
            'Difference between benefit and cost utilities'
        )

        try:
            ri = np.e ** diff
        except Exception as e:
            raise RuntimeError(f"Failed to compute preference scores: {e}") from e

        self._log_step(
            self.logger,
            'Preference scores (R)',
            np.array(ri, dtype=float),
            'Final AROMAN preference values — higher is better'
        )
        return ri

    def rank(self, defuzzify: callable = graded_mean_average_defuzzification):
        """
        Calculate rankings for S, R, and Q.

        Must be called after ``__call__()``.

        Parameters
        ----------
        defuzzify : callable, default=graded_mean_average_defuzzification
            Function used to transform fuzzy preference values into crisp
            scores.

        Returns
        -------
            tuple of ndarray
                (rank_S, rank_R, rank_Q), each of shape (m,). Lower
                values of S/R/Q receive rank 1 (best).

        Raises
        ------
            AttributeError
                If called before the method has been evaluated.
        """
        if defuzzify is not None and not callable(defuzzify):
            raise TypeError("'defuzzify' must be callable or None.")
        
        if self.preferences is None:
            raise AttributeError(f'{self.method_name}: call the method first before requesting a ranking.')

        crisp_prefs = np.array([defuzzify(tfn) for tfn in self.preferences])
        return rank_alternatives(crisp_prefs, descending=self._descending, method="average")
