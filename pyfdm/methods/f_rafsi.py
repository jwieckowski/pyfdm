# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger 
from ..validator import Validator
from ..utils import minmax_normalization, graded_mean_average_defuzzification

class fRAFSI(BaseFuzzyMethod):
    """
    Fuzzy RAFSI — Ranking of Alternatives through Functional mapping
    of criterion Sub-Intervals into a Single Interval.

    Every criterion's TFN values are linearly standardized into a common
    interval ``[lower_bound, upper_bound]`` (paper default ``[1, 6]``) using a per-criterion
    ideal and anti-ideal bound. These standardized values
    are then normalized into ``gamma`` in ``[0, 1]`` via the arithmetic
    mean A and harmonic mean H of ``lower_bound``/``upper_bound``:
    benefit criteria use a direct elementwise ratio (``phi / (2A)``), while
    cost criteria use a reciprocal relationship with the TFN components
    crossed (to keep the result validly ordered, ``l <= m <= u``, since
    fuzzy division reverses the interval). This orients `gamma` so that
    HIGHER is always better, for both benefit and cost criteria. Each
    alternative's fuzzy criteria function is the weighted (crisp weights)
    sum of its `gamma` values, defuzzified via the PERT-style
    expected value ``(l + 4m + u) / 6``. Higher scores indicate
    better alternatives.

    .. rubric:: Reference

        Bozanic, D., Milic, A., Tesic, D., Salabun, W., & Pamucar, D. (2021).
        D numbers - FUCOM - Fuzzy RAFSI model for selecting the group of
        construction machines for enabling mobility. Facta Universitatis,
        Series: Mechanical Engineering, 19(3), 447-471.

    Parameters
    ----------
    normalization : callable, default=minmax_normalization
        Function used to normalize the TFN decision matrix.
    defuzzify : callable, default=graded_mean_average_defuzzification
        Function used to transform the final fuzzy preference values into
        crisp scores.
    lower_bound : float, default=1.0
        Lower bound of the common mapping interval used in Step 3.
    upper_bound : float, default=6.0
        Upper bound of the common mapping interval used in Step 3. The
        paper recommends an ``lower_bound:upper_bound`` ratio of ``1:6``.
    ideal : np.ndarray | None, shape (n,), optional
        Crisp, analyst/expert-specified ideal (best) bound per criterion.
        If None (default), derived from the matrix (see class Notes).
    anti_ideal : np.ndarray | None, shape (n,), optional
        Crisp, analyst/expert-specified anti-ideal (worst) bound per
        criterion. If None (default), derived from the matrix (see class
        Notes).
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """

    _crisp_weights_required = True

    def __init__(
        self,
        normalization: callable=minmax_normalization, 
        defuzzify: callable=graded_mean_average_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy RAFSI method object.

        Parameters
        ----------
        normalization : callable, default=minmax_normalization
            Function used to normalize the TFN decision matrix.
        defuzzify : callable, default=graded_mean_average_defuzzification
            Function used to transform the final fuzzy preference values into
            crisp scores.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.

        Raises
        ------
        ValueError
            If `lower_bound >= upper_bound`, or `ideal`/`anti_ideal` are given with invalid
            shapes.
        """

        super().__init__(logger=logger)
        self.normalization = normalization
        self.defuzzify = defuzzify

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        lower_bound: float,
        upper_bound: float,
        ideal: np.ndarray,
        anti_ideal: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy RAFSI inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        lower_bound : float
            Lower mapping-interval bound in effect.
        upper_bound : float
            Upper mapping-interval bound in effect.
        ideal : np.ndarray | None
            Ideal bounds in effect (as passed at construction, or updated by
            the most recent call).
        anti_ideal : np.ndarray | None
            Anti-ideal bounds in effect (as passed at construction, or updated
            by the most recent call).

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
            Validator.validate_rafsi_input(
                matrix,
                lower_bound,
                upper_bound,
                ideal,
                anti_ideal
            )

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fRAFSI input: {e}") from e

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        lower_bound: float = 1.0,
        upper_bound: float = 6.0,
        ideal: np.ndarray | None = None,
        anti_ideal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy RAFSI method.

        This override exists to surface RAFSI's tuning parameters (`lower_bound`,
        `upper_bound`, `ideal`, `anti_ideal`) in the call signature/docstring for
        IDEs and `help()`; all actual validation, computation, and logging
        happen in `BaseFuzzyMethod.__call__` / `fRAFSI._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, crisp or fuzzy; defuzzified and normalized
            to sum to 1 via `normalize_weights`.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        lower_bound : float, default=1.0
            Lower bound of the common mapping interval.
        upper_bound : float, default=6.0
            Upper bound of the common mapping interval.
        ideal : np.ndarray | None, shape (n,), optional
            Crisp ideal bound per criterion. Overridable per-call.
        anti_ideal : np.ndarray | None, shape (n,), optional
            Crisp anti-ideal bound per criterion. Overridable per-call.

        Returns
        -------
        np.ndarray, shape (m,)
            Fuzzy RAFSI preference score for each alternative; higher is
            better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent, or `lower_bound`/`upper_bound`/`ideal`/
            `anti_ideal` fail validation.
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
        >>> ideal = np.array([65, 6, 100, 1, 7, 15])
        >>> anti_ideal = np.array([15, 1, 50, 7, 1, 1])
        >>> frafsi = fRAFSI()
        >>> scores = frafsi(matrix, weights, types, ideal=ideal, anti_ideal=anti_ideal)

        """

        return super().__call__(
            matrix,
            weights,
            types,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            ideal=ideal,
            anti_ideal=anti_ideal,
            *args,
            **kwargs
        )

    @staticmethod
    def _standardize(x: np.ndarray, lo: float, hi: float, lower_bound: float, upper_bound: float) -> np.ndarray:
        """
        Standardizes a raw TFN into the common interval [lower_bound, upper_bound] (Eq. 13).

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            Raw TFN value (l, m, u).
        lo : float
            The smaller of the criterion's ideal/anti-ideal bounds.
        hi : float
            The larger of the criterion's ideal/anti-ideal bounds.
        lower_bound : float
            Lower bound of the mapping interval.
        upper_bound : float
            Upper bound of the mapping interval.

        Returns
        -------
        np.ndarray, shape (3,)
            Standardized TFN (phi).
        """
        return lower_bound + (upper_bound - lower_bound) * (x - lo) / (hi - lo)

    @staticmethod
    def _normalize_gamma(phi: np.ndarray, is_benefit: bool, lower_bound: float, upper_bound: float) -> np.ndarray:
        """
        Normalizes a standardized TFN into gamma in [0, 1] (Eqs. 16-18).

        For benefit criteria, applies a direct elementwise ratio. For cost
        criteria, applies a reciprocal relationship with the TFN
        components crossed, to keep the result validly ordered
        (fuzzy division reverses the interval).

        Parameters
        ----------
        phi : np.ndarray, shape (3,)
            Standardized TFN value.
        is_benefit : bool
            Whether the criterion is a benefit criterion (True) or a cost
            criterion (False).
        lower_bound : float
            Lower bound of the mapping interval.
        upper_bound : float
            Upper bound of the mapping interval.

        Returns
        -------
        np.ndarray, shape (3,)
            Normalized TFN (gamma), oriented so higher is always better.
        """
        A = (lower_bound + upper_bound) / 2.0
        H = 2.0 / (1.0 / lower_bound + 1.0 / upper_bound)

        if is_benefit:
            return phi / (2 * A)

        l, m, u = phi
        return np.array([H / (2 * u), H / (2 * m), H / (2 * l)])

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        lower_bound: float = 1.0,
        upper_bound: float = 6.0,
        ideal: np.ndarray | None = None,
        anti_ideal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        RAFSI method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, crisp or fuzzy; defuzzified and normalized
            to sum to 1.
        types : np.ndarray, shape (n,)
            Criteria types (``1`` for benefit, ``-1`` for cost).
        lower_bound : float, default=1.0
            Lower bound of the common mapping interval.
        upper_bound : float, default=6.0
            Upper bound of the common mapping interval.
        ideal : np.ndarray | None, shape (n,), optional
            Crisp ideal bound per criterion. Overridable per-call.
        anti_ideal : np.ndarray | None, shape (n,), optional
            Crisp anti-ideal bound per criterion. Overridable per-call.

        Returns
        -------
        np.ndarray, shape (m,)
            Final RAFSI preference score for each alternative. Higher
            values indicate better alternatives.

        Raises
        ------
        RuntimeError
            If the bounds derivation, standardization, normalization,
            weighting, or defuzzification steps fail unexpectedly.
        """

        m, n, _ = matrix.shape

        # Ideal / anti-ideal bounds per criterion
        try:
            if ideal is None or anti_ideal is None:
                calculated_ideal = np.zeros(n)
                calculated_anti = np.zeros(n)

                for j in range(n):
                    col_max = np.max(matrix[:, j, 2])
                    col_min = np.min(matrix[:, j, 2])

                    if types[j] == 1:
                        calculated_ideal[j] = col_max
                        calculated_anti[j] = col_min
                    else:
                        calculated_ideal[j] = col_min
                        calculated_anti[j] = col_max

                if ideal is None:
                    ideal = calculated_ideal

                if anti_ideal is None:
                    anti_ideal = calculated_anti

        except Exception as e:
            raise RuntimeError(f"Failed to derive ideal/anti-ideal bounds: {e}") from e
            
        self._log_step(self.logger, 'Ideal bounds', ideal, 'Crisp ideal (best) bound per criterion')
        self._log_step(self.logger, 'Anti-ideal bounds', anti_ideal, 'Crisp anti-ideal (worst) bound per criterion')

        # Standardize into [lower_bound, upper_bound] 
        try:
            phi = np.zeros((m, n, 3))
            for j in range(n):
                lo = min(ideal[j], anti_ideal[j])
                hi = max(ideal[j], anti_ideal[j])
                for i in range(m):
                    phi[i, j] = self._standardize(matrix[i, j], lo, hi, lower_bound, upper_bound)
        except Exception as e:
            raise RuntimeError(f"Failed to standardize the decision matrix (phi): {e}") from e

        self._log_step(self.logger, 'Standardized matrix (phi)', phi, 'Eq. 13: linear mapping of each TFN into [lower_bound, upper_bound]')

        # Normalize into gamma in [0, 1]
        try:
            gamma = np.zeros((m, n, 3))
            for j in range(n):
                for i in range(m):
                    gamma[i, j] = self._normalize_gamma(phi[i, j], types[j] == 1, lower_bound, upper_bound)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize the standardized matrix (gamma): {e}") from e

        self._log_step(self.logger, 'Normalized matrix (gamma)', gamma, 'Eqs. 16-18: normalized so higher is always better, for both benefit and cost criteria')

        try:
            Q_fuzzy = np.sum(gamma * weights[np.newaxis, :, np.newaxis], axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to compute the fuzzy criteria function (Q): {e}") from e

        self._log_step(self.logger, 'Fuzzy criteria function (Q)', Q_fuzzy, 'Eq. 19: weighted sum of gamma across criteria')

        try:
            scores = np.array([self.defuzzify(q) for q in Q_fuzzy])
        except Exception as e:
            raise RuntimeError(f"Defuzzification failed: {e}") from e

        self._log_step(self.logger, 'Preference scores (Q, defuzzified)', scores, 'Eq. 20: PERT-style expected value — higher is better')
        return scores