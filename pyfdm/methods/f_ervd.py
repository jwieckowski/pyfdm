# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..validator import Validator
from ..utils import saw_normalization, vertex_distance

class fERVD(BaseFuzzyMethod):
    """
    Fuzzy ERVD — Election based on Relative Value Distances.

    Measures each alternative's relative closeness to a positive/negative
    ideal solution derived from a prospect-theory value function applied to
    the (normalized) distance between every criterion value and a
    reference (aspiration) point. If no reference point is provided, the
    positive ideal solution (column-wise max for benefit criteria, min for
    cost criteria) is used instead. Gains relative to the reference are
    valued with diminishing sensitivity (`_alpha`), while losses are
    amplified by a loss-aversion coefficient (`lambda_`), following
    Kahneman & Tversky's prospect theory. Higher scores indicate better
    alternatives.

    .. rubric:: Reference

        Shojaeimehr, S., & Rahmani, D. (2022). Risk management of photovoltaic 
        power plants using a novel fuzzy multi-criteria decision-making method 
        based on prospect theory: A sustainable development approach. 
        Energy Conversion and Management: X, 16, 100293.

    Parameters
    ----------
    normalization : callable, default=saw_normalization
        Function used to normalize the TFN decision matrix.
    distance : callable, default=vertex_distance
        Function computing the distance between two individual TFNs
        (called pairwise, not on whole arrays).
    ref_point : np.ndarray | None, shape (n, 3), optional
        TFN reference (aspiration) point per criterion. If None (default),
        the positive ideal solution derived from the matrix is used.
    _alpha : float, default=0.88
        Prospect theory diminishing-sensitivity exponent for gains/losses
        (0 < alpha <= 1).
    _lambda : float, default=2.25
        Prospect theory loss-aversion coefficient (lambda >= 1); losses are
        weighted `lambda` times more heavily than equivalent gains.
    d : float, default=0.4
        Compromise parameter blending the negative-ideal and positive-ideal
        relative distances in the final score.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    """

    _descending = True

    def __init__(
        self,
        normalization: callable = saw_normalization,
        distance: callable = vertex_distance,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy ERVD method object.

        Parameters
        ----------
        normalization : Callable, default=saw_normalization
            Function used to normalize the TFN decision matrix.
        distance : Callable, default=vertex_distance
            Function computing the distance between two individual TFNs.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.

        Raises
        ------
        ValueError
            If `d` is not in [0, 1], or `ref_point` is given with an invalid shape.
        """
        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance = distance

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        ref_point: np.ndarray | None = None,
        _alpha: float = 0.88,
        _lambda: float = 2.25,
        d: float = 0.4,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy ERVD method.

        This override exists purely to surface ERVD's specific tuning
        parameters (`ref_point`, `_alpha`, `_lambda`, `d`) in the call
        signature/docstring for IDEs and `help()`; all actual validation,
        computation, and logging happen in `BaseFuzzyMethod.__call__` /
        `fERVD._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        ref_point : np.ndarray | None, shape (n, 3), optional
            TFN reference (aspiration) point per criterion. If None
            (default), the positive ideal solution derived from `matrix`
            is used (column-wise max for benefit criteria, min for cost
            criteria).
        _alpha : float, default=0.88
            Prospect theory diminishing-sensitivity exponent for
            gains/losses (must be in ``(0, 1]``).
        _lambda : float, default=2.25
            Prospect theory loss-aversion coefficient (must be
            ``>= 1``); losses are weighted `_lambda` times more heavily
            than equivalent gains.
        d : float, default=0.4
            Compromise parameter in ``[0, 1]`` blending the negative-ideal
            and positive-ideal relative distances in the final score.

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy ERVD preference score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent, or `_alpha`/`_lambda`/`d`/
            `ref_point` fail validation (see `_calculate`).
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
        >>> fervd = fERVD()
        >>> scores = fervd(matrix, weights, types, d=0.7, _lambda=2.0)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            ref_point=ref_point,
            _alpha=_alpha,
            _lambda=_lambda,
            d=d,
            *args,
            **kwargs
        )

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        ref_point: np.ndarray | None,
        _alpha: float,
        _lambda: float,
        d: float,
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
        ref_point : np.ndarray | None, shape (n, 3), optional
            TFN reference (aspiration) point per criterion. If None
            (default), the positive ideal solution derived from `matrix`
            is used (column-wise max for benefit criteria, min for cost
            criteria).
        _alpha : float, default=0.88
            Prospect theory diminishing-sensitivity exponent for
            gains/losses (must be in ``(0, 1]``).
        _lambda : float, default=2.25
            Prospect theory loss-aversion coefficient (must be
            ``>= 1``); losses are weighted `_lambda` times more heavily
            than equivalent gains.
        d : float, default=0.6
            Compromise parameter in ``[0, 1]`` blending the negative-ideal
            and positive-ideal relative distances in the final score.

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
            Validator.validate_param_range(d, 0, 1, 'd')
            Validator.validate_ervd_input(ref_point, matrix.shape[1])

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fERVD input: {e}") from e

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        ref_point: np.ndarray | None = None,
        _alpha: float = 0.88,
        _lambda: float = 2.25,
        d: float = 0.4,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        ERVD method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,), broadcast to TFN
            (w, w, w)) or already fuzzy (shape (n, 3)).
        types : np.ndarray, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        ref_point : np.ndarray | None, shape (n, 3), optional
            TFN reference (aspiration) point per criterion. Defaults to the
            positive ideal solution derived from the matrix.
        _alpha : float, default=0.88
            Prospect theory diminishing-sensitivity exponent
            (0 < alpha <= 1).
        _lambda : float, default=2.25
            Prospect theory loss-aversion coefficient (lambda >= 1).
        d : float, default=0.4
            Compromise parameter in [0, 1] blending negative-ideal and
            positive-ideal relative distances.

        Returns
        -------
        np.ndarray, shape (m,)
            ERVD preference score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes, or if
            `ref_point` was set but its shape doesn't match
            `matrix`'s number of criteria.
        RuntimeError
            If normalization, the value-function computation, or the final
            score aggregation fails unexpectedly.
        """

        m, n, _ = matrix.shape

        # Compute reference point
        try:
            if ref_point is None:
                ref = np.zeros((n, 3))
                for j in range(n):
                    if types[j] == 1:
                        ref[j] = np.max(matrix[:, j, :], axis=0)
                    else:
                        ref[j] = np.min(matrix[:, j, :], axis=0)
            else:
                ref = np.array(ref_point, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute the reference point: {e}") from e

        self._log_step(self.logger, 'Reference point', ref, 'Aspiration-level TFN per criterion')

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Result of applying the normalization function')

        try:
            maxx = np.max(np.max(matrix, axis=0), axis=1)
            nref = [ref[j] / maxx[j] for j in range(n)]
        except Exception as e:
            raise RuntimeError(f"Failed to normalize the reference point: {e}") from e

        self._log_step(self.logger, 'Normalized reference point', np.array(nref), 'Reference point scaled the same way as the decision matrix')

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        # Prospect-theory value function
        try:
            vnmatrix = np.zeros(matrix.shape, dtype=float)
            for j in range(n):
                w = weights[j]

                for i in range(m):
                    val = nmatrix[i, j]
                    r = nref[j]

                    dist = self.distance(val, r)

                    value = np.zeros(3)
                    loss = False

                    if types[j] == 1:     
                        for k in range(3):
                            if val[k] >= r[k]:
                                value[k] = w[k] * (dist ** _alpha)
                            else:
                                value[k] = -_lambda * w[k] * (dist ** _alpha)
                                loss = True
                    else:                 
                        for k in range(3):
                            if val[k] <= r[k]:
                                value[k] = w[k] * (dist ** _alpha)
                            else:
                                value[k] = -_lambda * w[k] * (dist ** _alpha)
                                loss = True

                    if loss:
                        value = value[::-1]

                    vnmatrix[i, j] = value
        except Exception as e:
            raise RuntimeError(f"Failed to compute the prospect-theory value matrix: {e}") from e

        self._log_step(self.logger, 'Value-weighted normalized matrix', vnmatrix, 'Prospect-theory value function applied to distances from the reference point')

        # Positive/negative ideal solutions
        try:
            v_plus = np.zeros((n, 3))
            v_minus = np.zeros((n, 3))

            for j in range(n):
                if types[j] == 1:
                    v_plus[j] = np.max(vnmatrix[:, j, :])
                    v_minus[j] = np.min(vnmatrix[:, j, :])
                else: 
                    v_plus[j] = np.min(vnmatrix[:, j, :])
                    v_minus[j] = np.max(vnmatrix[:, j, :])
        except Exception as e:
            raise RuntimeError(f"Failed to compute positive/negative ideal solutions: {e}") from e

        self._log_step(self.logger, 'Positive ideal solution (v+)', v_plus, 'Componentwise best value per criterion')
        self._log_step(self.logger, 'Negative ideal solution (v-)', v_minus, 'Componentwise worst value per criterion')

        # Distances to ideal solutions
        try:
            S_plus = np.zeros(m)
            S_minus = np.zeros(m)

            for i in range(m):
                for j in range(n):
                    S_plus[i] += self.distance(vnmatrix[i, j], v_plus[j])
                    S_minus[i] += self.distance(vnmatrix[i, j], v_minus[j])

        except Exception as e:
            raise RuntimeError(f"Failed to compute distances to ideal solutions: {e}") from e

        self._log_step(self.logger, 'Distance to positive ideal (S+)', S_plus, 'Sum of per-criterion distances to v+')
        self._log_step(self.logger, 'Distance to negative ideal (S-)', S_minus, 'Sum of per-criterion distances to v-')

        # Final relative closeness score
        try:
            S_plus_sum = np.sum(S_plus)
            S_minus_sum = np.sum(S_minus)

            S_plus_sum = S_plus_sum if S_plus_sum != 0 else 1e-10
            S_minus_sum = S_minus_sum if S_minus_sum != 0 else 1e-10

            p = d * (S_minus / S_minus_sum) - (1 - d) * (S_plus / S_plus_sum)
        except Exception as e:
            raise RuntimeError(f"Failed to compute final preference scores: {e}") from e

        self._log_step(self.logger, 'Preference scores (p)', p, f'Relative closeness score (d={d}) — higher is better')

        return p
