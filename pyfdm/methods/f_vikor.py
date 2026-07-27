# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..validator import Validator
from ..utils import mean_area_defuzzification, rank_alternatives

class fVIKOR(BaseFuzzyMethod):
    """
    Fuzzy VIseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR)
    method .

    The method determines a compromise ranking by simultaneously considering
    the overall group utility and the maximum individual regret. After
    (optionally) normalizing the fuzzy decision matrix, fuzzy ideal and
    nadir solutions are identified for every criterion. Each alternative's
    weighted normalized distance from the ideal solution is then used to
    compute three measures:

    * S - group utility measure,
    * R - individual regret measure,
    * Q - compromise ranking index.

    The parameter ``v`` controls the balance between maximizing group utility
    (S) and minimizing individual regret (R). Lower values of all three
    measures indicate better alternatives.
    
    .. rubric:: Reference
        Opricovic, S. (2007). A fuzzy compromise solution for multicriteria
        problems. International Journal of Uncertainty, Fuzziness and 
        Knowledge-Based Systems, 15(03), 363-380.

    Parameters
    ----------
    defuzzify : callable, default=mean_area_defuzzification
        Function used to transform fuzzy preference values into crisp
        scores.
    normalization : callable, optional, default=None
        Function used to normalize the TFN decision matrix.
    v : float, default=0.5
        Compromise coefficient balancing group utility (S) and
        individual regret (R). Must lie in the interval [0, 1].
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    """

    _descending = False 

    def __init__(
        self,
        defuzzify: callable = mean_area_defuzzification,
        normalization: callable = None,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy VIKOR method object.

        Parameters
        ----------
        defuzzify : callable, default=mean_area_defuzzification
            Function used to transform fuzzy preference values into crisp
            scores.
        normalization : callable, optional, default=None
            Function used to normalize the TFN decision matrix.
        logger : StepLogger | None, optional
            Optional logger used for recording intermediate computation
            steps.
        """
        super().__init__(logger=logger)
        self.defuzzify = defuzzify
        self.normalization = normalization

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        v: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy VIKOR inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        v : float, default=0.5
            Compromise coefficient balancing group utility (S) and
            individual regret (R). Must lie in the interval [0, 1].

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
            Validator.validate_param_range(v, 0, 1, 'v')

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fVIKOR input: {e}") from e


    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        v: float = 0.5, 
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy VIKOR method.

        This override exists to give `fVIKOR` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fVIKOR._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        v : float, default=0.5
            Compromise coefficient balancing group utility (S) and
            individual regret (R). Must lie in the interval [0, 1].

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:

            * S - group utility measure,
            * R - individual regret measure,
            * Q - compromise ranking index.

            Fuzzy VIKOR preference score for each alternative.
            Lower values indicate better alternatives.

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
        >>> fvikor = fVIKOR()
        >>> scores = fvikor(matrix, weights, types)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            v=v,
            *args,
            **kwargs
        )
    
    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        v: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes preference measures for each alternative using the fuzzy
        VIKOR method.

        The decision matrix is optionally normalized, after which fuzzy ideal
        and nadir solutions are determined for every criterion. Weighted
        normalized distances from the ideal solution are then used to calculate
        the group utility measure (S), individual regret measure (R), and the
        compromise ranking index (Q). Finally, all three fuzzy measures are
        defuzzified.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp or fuzzy. Crisp weights are
            automatically converted to triangular fuzzy weights.
        types : np.ndarray, shape (n,)
            Criteria types (1 for benefit criteria, -1 for cost criteria).
        v : float, default=0.5
            Compromise coefficient balancing group utility and individual
            regret.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the defuzzified S, R and Q vectors.

        Raises
        ------
        RuntimeError
            If normalization, ideal/nadir determination, distance
            computation, aggregation, defuzzification or compromise
            score calculation fails unexpectedly.
        """

        try:
            if self.normalization is not None:
                nmatrix = self.normalization(matrix, types)
            else:
                nmatrix = matrix.copy()
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(
            self.logger,
            "Normalized matrix",
            nmatrix,
            "Result of applying normalization (or original matrix when normalization=None)"
        )

        try:
            ideal = np.zeros((nmatrix.shape[1], 3))
            nadir = np.zeros((nmatrix.shape[1], 3))

            for j in range(nmatrix.shape[1]):
                if types[j] == 1:
                    ideal[j] = np.max(nmatrix[:, j], axis=0)
                    nadir[j] = np.min(nmatrix[:, j], axis=0)
                else:
                    ideal[j] = np.min(nmatrix[:, j], axis=0)
                    nadir[j] = np.max(nmatrix[:, j], axis=0)

        except Exception as e:
            raise RuntimeError(f"Failed to determine the ideal and nadir solutions: {e}") from e

        self._log_step(
            self.logger,
            "Ideal solution (F*)",
            ideal,
            "Best value for each criterion"
        )

        self._log_step(
            self.logger,
            "Nadir solution (F-)",
            nadir,
            "Worst value for each criterion"
        )

        try:
            d = np.zeros(nmatrix.shape)

            for i in range(nmatrix.shape[0]):
                for j in range(nmatrix.shape[1]):
                    if types[j] == 1:
                        d[i, j] = (
                            ideal[j] - np.flipud(nmatrix[i, j])
                        ) / (ideal[j, 2] - nadir[j, 0])
                    else:
                        d[i, j] = (
                            nmatrix[i, j] - np.flipud(ideal[j])
                        ) / (nadir[j, 2] - ideal[j, 0])

        except Exception as e:
            raise RuntimeError(f"Failed to compute the normalized distance matrix: {e}") from e

        self._log_step(
            self.logger,
            "Distance matrix (D)",
            d,
            "Normalized distance from the ideal solution"
        )

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            weighted_d = d * weights

            S = np.zeros((nmatrix.shape[0], 3))
            R = np.zeros((nmatrix.shape[0], 3))

            for i in range(nmatrix.shape[0]):
                S[i] = np.sum(weighted_d[i], axis=0)
                R[i] = np.max(weighted_d[i], axis=0)

        except Exception as e:
            raise RuntimeError(f"Failed to compute the S and R measures: {e}") from e

        self._log_step(
            self.logger,
            "Utility measure (S)",
            S,
            "Weighted sum of normalized distances"
        )

        self._log_step(
            self.logger,
            "Regret measure (R)",
            R,
            "Maximum weighted normalized distance"
        )

        try:
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            R_min = np.min(R, axis=0)
            R_max = np.max(R, axis=0)

            Q = np.zeros((nmatrix.shape[0], 3))

            for i in range(nmatrix.shape[0]):
                Q[i] = (
                    v * (S[i] - np.flipud(S_min)) / (S_max[2] - S_min[0]) +
                    (1 - v) * (R[i] - np.flipud(R_min)) / (R_max[2] - R_min[0])
                )

        except Exception as e:
            raise RuntimeError(f"Failed to compute the compromise measure (Q): {e}") from e

        self._log_step(
            self.logger,
            "Compromise measure (Q)",
            Q,
            f"Compromise index computed using v = {v}"
        )

        try:
            crisp_S = np.array([self.defuzzify(s) for s in S])
            crisp_R = np.array([self.defuzzify(r) for r in R])
            crisp_Q = np.array([self.defuzzify(q) for q in Q])

        except Exception as e:
            raise RuntimeError(f"Failed to defuzzify the VIKOR measures: {e}") from e

        self._log_step(
            self.logger,
            "Defuzzified S",
            crisp_S,
            "Group utility measure — lower is better"
        )

        self._log_step(
            self.logger,
            "Defuzzified R",
            crisp_R,
            "Individual regret measure — lower is better"
        )

        self._log_step(
            self.logger,
            "Defuzzified Q",
            crisp_Q,
            f"Compromise ranking index (v = {v}) — lower is better"
        )

        return crisp_S, crisp_R, crisp_Q

    def rank(self):
        """
        Calculate rankings for S, R, and Q.

        Must be called after ``__call__()``.

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
        if self.preferences is None:
            raise AttributeError(f'{self.method_name}: call the method first before requesting a ranking.')
        
        S, R, Q = self.preferences
        return (
            rank_alternatives(S, descending=False, method="average"),
            rank_alternatives(R, descending=False, method="average"),
            rank_alternatives(Q, descending=False, method="average"),
        )
