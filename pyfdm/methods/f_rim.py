# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger 
from ..validator import Validator
from ..utils import vertex_distance

class fRIM(BaseFuzzyMethod):
    """
    Fuzzy RIM — Reference Ideal Method.

    Every criterion has, in addition to its universe-of-discourse bounds
    (`lower_bound`, `upper_bound`), a "reference ideal" sub-interval
    (`lower_reference`, `upper_reference`) representing the most
    desirable range of values for that criterion (which may be narrower
    than, and need not be centred within, the full bounds). Each TFN
    value is compared against these bounds and reference interval (via a
    TFN vertex distance) and mapped to a crisp normalized degree in
    [0, 1]: 1.0 if it falls within the reference interval, and a
    proportionally decreasing value the further it lies outside that
    interval (toward `lower_bound` or `upper_bound`). Weighted normalized
    values are then compared to the fuzzy positive ideal (matching the
    reference interval exactly) and the negative ideal (zero), and the
    final preference score is a TOPSIS-style relative closeness:
    ``p_i = i_minus_i / (i_plus_i + i_minus_i)``. Higher scores indicate
    better alternatives.

    Parameters
    ----------
    lower_bound : np.ndarray | None, shape (n, 3), optional
        Lower bound of each criterion's universe of discourse (TFN).
        Overridable per-call.
    upper_bound : np.ndarray | None, shape (n, 3), optional
        Upper bound of each criterion's universe of discourse (TFN).
        Overridable per-call.
    lower_reference : np.ndarray | None, shape (n, 3), optional
        Lower bound of each criterion's reference ideal interval (TFN).
        Overridable per-call.
    upper_reference : np.ndarray | None, shape (n, 3), optional
        Upper bound of each criterion's reference ideal interval (TFN).
        Overridable per-call.
    distance : callable, default=vertex_distance
        Function computing the distance between two individual TFNs, used
        both to build the normalized matrix and (implicitly, via squared
        Euclidean aggregation) in the final relative-closeness score.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Cables, E., Lamata, M. T., & Verdegay, J. L. (2017). 
    FRIM—fuzzy reference ideal method in multicriteria 
    decision making. In Soft computing applications for 
    group decision-making and consensus modeling 
    (pp. 305-317). Cham: Springer International Publishing.
    """
    _crisp_weights_required = True

    def __init__(
        self,
        distance: callable = vertex_distance,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy RIM method object.

        Parameters
        ----------
        distance : callable, default=vertex_distance
            Function computing the distance between two individual TFNs.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.

        Raises
        ------
        ValueError
            If any of the given bounds have an invalid shape.
        """
        super().__init__(logger=logger)
        self.distance = distance

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        lower_reference: np.ndarray,
        upper_reference: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy RIM inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        lower_bound : np.ndarray
            The universe-of-discourse lower bound in effect (as passed at
            construction, or updated by the most recent call).
        upper_bound : np.ndarray
            The universe-of-discourse upper bound in effect.
        lower_reference : np.ndarray
            The reference-interval lower bound in effect.
        upper_reference : np.ndarray
            The reference-interval upper bound in effect.

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
            Validator.validate_rim_input(
                matrix, 
                lower_bound,
                upper_bound,
                lower_reference,
                upper_reference
            )

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fRIM input: {e}") from e


    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        lower_bound: np.ndarray | None = None,
        upper_bound: np.ndarray | None = None,
        lower_reference: np.ndarray | None = None,
        upper_reference: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy RIM method.

        This override exists to surface RIM's domain/reference bounds in
        the call signature/docstring for IDEs and `help()`; all actual
        validation, computation, and logging happen in
        `BaseFuzzyMethod.__call__` / `fRIM._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,)
            Crisp criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types (currently unused directly by RIM's own
            formula, since desirability is fully encoded by the reference
            interval; forwarded for interface consistency).
        lower_bound : np.ndarray | None, optional, default=None
            The universe-of-discourse lower bound in effect (as passed at
            construction, or updated by the most recent call).
        upper_bound : np.ndarray | None, optional, default=None
            The universe-of-discourse upper bound in effect.
        lower_reference : np.ndarray | None, optional, default=None
            The reference-interval lower bound in effect.
        upper_reference : np.ndarray | None, optional, default=None
            The reference-interval upper bound in effect.

        Returns
        -------
        np.ndarray, shape (m,)
            Fuzzy RIM preference score for each alternative; higher is
            better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent, if any bound is missing
            (neither passed here nor set at construction), or if bounds
            fail validation.
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
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> types = np.array([1, 1, 1])
        >>> lower_b = np.array([
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ...     [1, 2, 3],
        ... ])
        >>> upper_b = np.array([
        ...     [8, 9, 10],
        ...     [8, 8, 8],
        ...     [9, 9, 9],
        ... ])
        >>> lower_ref = np.array([
        ...     [3, 4, 5],
        ...     [2, 3, 4],
        ...     [2, 3, 4],
        ... ])
        >>> upper_ref = np.array([
        ...     [6, 7, 8],
        ...     [5, 6, 7],
        ...     [5, 6, 7],
        ... ])
        >>> frim = fRIM()
        >>> scores = frim(
        ...     matrix,
        ...     weights,
        ...     types,
        ...     lower_bound=lower_b,
        ...     upper_bound=upper_b,
        ...     lower_reference=lower_ref,
        ...     upper_reference=upper_ref,
        ... )
        """
        return super().__call__(
            matrix,
            weights,
            types,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_reference=lower_reference,
            upper_reference=upper_reference,
            *args,
            **kwargs
        )

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        lower_reference: np.ndarray,
        upper_reference: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        RIM method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,)
            Crisp criterion weights.
        types : np.ndarray, shape (n,)
            Criteria types (unused directly; see `__call__`).
        lower_bound : np.ndarray
            The universe-of-discourse lower bound in effect (as passed at
            construction, or updated by the most recent call).
        upper_bound : np.ndarray
            The universe-of-discourse upper bound in effect.
        lower_reference : np.ndarray
            The reference-interval lower bound in effect.
        upper_reference : np.ndarray
            The reference-interval upper bound in effect.

        Returns
        -------
        np.ndarray, shape (m,)
            Final RIM preference score for each alternative. Higher values
            indicate better alternatives.

        Raises
        ------
        ValueError
            If any required bound is missing.
        RuntimeError
            If normalization or the final aggregation step encounters a
            value outside the expected [lower_bound, upper_bound] range,
            or fails unexpectedly for any other reason.
        """

        m, n, _ = matrix.shape

        # Automatically derive missing TFN bounds/reference points
        try:
            if lower_bound is None:
                lower_bound = np.min(matrix, axis=0)

            if upper_bound is None:
                upper_bound = np.max(matrix, axis=0)

            if lower_reference is None:
                lower_reference = np.percentile(matrix, 45, axis=0)

            if upper_reference is None:
                upper_reference = np.percentile(matrix, 55, axis=0)

        except Exception as e:
            raise RuntimeError(f"Failed to derive fuzzy RIM bounds and references: {e}") from e

        self._log_step(
            self.logger,
            "Lower fuzzy bounds",
            lower_bound,
            "Minimum TFN values for each criterion"
        )

        self._log_step(
            self.logger,
            "Upper fuzzy bounds",
            upper_bound,
            "Maximum TFN values for each criterion"
        )

        self._log_step(
            self.logger,
            "Lower reference points",
            lower_reference,
            "Lower reference TFN values for each criterion (given explicitly or calculated as 45th percentile of decision matrix)"
        )

        self._log_step(
            self.logger,
            "Upper reference points",
            upper_reference,
            "Upper reference TFN values for each criterion (given explicitly or calculated as 55th percentile of decision matrix)"
        )

        # Normalize each TFN cell into a crisp degree in [0,1]
        try:
            nmatrix = np.zeros((m, n), dtype=float)
            for i in range(m):
                for j in range(n):
                    x = matrix[i, j]
                    l_ref, u_ref = lower_reference[j], upper_reference[j]
                    l_b, u_b = lower_bound[j], upper_bound[j]

                    if np.all(l_ref <= x) and np.all(x <= u_ref):
                        nmatrix[i, j] = 1.0

                    elif np.all(l_b <= x) and np.all(x <= u_ref) and self.distance(l_b, l_ref) != 0:
                        dist_min = min(self.distance(x, l_ref), self.distance(x, u_ref))
                        dist_den = self.distance(l_b, l_ref)
                        nmatrix[i, j] = 1 - (dist_min / dist_den)

                    elif np.all(u_ref <= x) and np.all(x <= u_b) and self.distance(u_ref, u_b) != 0:
                        dist_min = min(self.distance(x, l_ref), self.distance(x, u_ref))
                        dist_den = self.distance(u_ref, u_b)
                        nmatrix[i, j] = 1 - (dist_min / dist_den)

                    else:
                        raise RuntimeError(
                            f"Alternative {i}, criterion {j} (value={x.tolist()}) "
                            f"does not fall within [lower_bound, upper_bound] "
                            f"relative to the reference interval; check that "
                            f"'lower_bound'/'upper_bound'/'lower_reference'/"
                            f"'upper_reference' correctly bound the decision matrix."
                        )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to compute the normalized matrix: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Crisp degree in [0, 1]: 1.0 within the reference interval, decreasing toward the bounds')

        # Weight and aggregate relative to the fuzzy ideals
        try:
            wnmatrix = nmatrix * weights
            i_plus = np.sqrt(np.sum((wnmatrix - weights) ** 2, axis=1))
            i_minus = np.sqrt(np.sum(wnmatrix ** 2, axis=1))
        except Exception as e:
            raise RuntimeError(f"Failed to compute distances to the positive/negative ideals: {e}") from e

        self._log_step(self.logger, 'Weighted normalized matrix', wnmatrix, 'Element-wise product of normalized matrix and weights')
        self._log_step(self.logger, 'Distance to positive ideal (I+)', i_plus, 'Distance from the weighted normalized values to the (fully in-reference) positive ideal')
        self._log_step(self.logger, 'Distance to negative ideal (I-)', i_minus, 'Distance from the weighted normalized values to the (zero) negative ideal')

        try:
            i_sum = i_plus + i_minus
            i_sum_safe = np.where(i_sum == 0, 1e-10, i_sum)
            p = i_minus / i_sum_safe
        except Exception as e:
            raise RuntimeError(f"Failed to compute the final preference scores: {e}") from e

        self._log_step(self.logger, 'Preference scores (p)', p, 'Relative closeness p = I- / (I+ + I-) — higher is better')
        return p