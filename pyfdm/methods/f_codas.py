# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import max_normalization
from ..utils import euclidean_distance, hamming_distance

class fCODAS(BaseFuzzyMethod):
    """
    Fuzzy COmbinative Distance-based ASsessment (CODAS) method.

    Normalizes the TFN decision matrix, applies criterion weights, derives
    a fuzzy negative (anti-ideal) solution as the column-wise minimum of
    the weighted normalized matrix, and computes two distance measures
    (`distance_1`, `distance_2`) from every alternative to that negative
    solution. A pairwise Relative Assessment matrix combines both
    distances -- the second distance only breaks ties/near-ties in the
    first, via a threshold function controlled by `tau` -- and each
    alternative's Assessment Score is the row sum of that matrix. Higher
    scores indicate better alternatives.

    Parameters
    ----------
    normalization : callable, default=max_normalization
        Function used to calculate the normalized decision matrix. 
    distance_1 : callable
        Primary distance function (default: Euclidean distance) used to
        compare each alternative to the fuzzy negative solution.
    distance_2 : callable
        Secondary distance function (default: Hamming distance) used the
        same way, and to break ties/near-ties in `distance_1`.
    tau : float
        Threshold parameter: `distance_2` only contributes to the
        Relative Assessment matrix when the corresponding `distance_1`
        difference has an absolute value of at least `tau`.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Panchal, D., Chatterjee, P., Shukla, R. K., Choudhury, T., & Tamosaitiene, J. (2017).
    Integrated Fuzzy AHP-Codas Framework for Maintenance Decision in Urea Fertilizer Industry.
    Economic Computation & Economic Cybernetics Studies & Research, 51(3).
    """

    def __init__(
        self,
        normalization=max_normalization,
        distance_1=euclidean_distance,
        distance_2=hamming_distance,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy CODAS method object with max normalization function
        and Euclidean and Hamming distance metrics.

        Parameters
        ----------
        normalization : callable, default=max_normalization
            Function used to calculate the normalized decision matrix.
        distance_1 : callable, default=euclidean_distance
            Function used to calculate distance from the fuzzy negative
            solution (primary distance measure).
        distance_2 : callable, default=hamming_distance
            Function used to calculate distance from the fuzzy negative
            solution (secondary/tie-breaking distance measure).
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """

        super().__init__(logger=logger)
        self.normalization = normalization
        self.distance_1 = distance_1
        self.distance_2 = distance_2

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        tau: float = 0.02,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy CODAS method.

        This override exists purely to surface CODAS's specific tuning
        parameters in the call signature/docstring for IDEs and `help()`; 
        all actual validation, computation, and logging happen in 
        `BaseFuzzyMethod.__call__` / `fCODAS._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        tau : float, default=0.02
            Threshold parameter controlling when the secondary distance
            contributes to the Relative Assessment matrix.

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy CODAS preference score for each alternative; higher is better.

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
        >>> fcodas = fCODAS()
        >>> scores = fcodas(matrix, weights, types)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            tau=tau,
            *args,
            **kwargs
        )

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        tau: float,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        CODAS method.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,), broadcast to TFN
            (w, w, w)) or already fuzzy (shape (n, 3)).
        types : np.ndarray, shape (n,)
            Criteria types (e.g. +1 for benefit, -1 for cost), forwarded to
            `self.normalization`.
        tau : float, default=0.02
            Threshold parameter controlling when the secondary distance
            contributes to the Relative Assessment matrix.

        Returns
        -------
        np.ndarray, shape (m,)
            Assessment score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If normalization, distance computation, or the assessment
            aggregation fails unexpectedly.
        """

        def _psi(x: float, tau: float = 0.02) -> int:
            """Threshold function: 1 if |x| >= tau, else 0."""
            return 1 if np.abs(x) >= tau else 0

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Result of applying the normalization function')

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wmatrix = nmatrix * weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights to the normalized matrix: {e}") from e

        self._log_step(self.logger, 'Weighted normalized matrix', wmatrix, 'Element-wise product of normalized matrix and weights')

        try:
            NS = np.min(wmatrix, axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute the fuzzy negative solution: {e}") from e

        self._log_step(self.logger, 'Fuzzy Negative Solution (NS)', NS, 'Column-wise minimum of the weighted normalized matrix')

        try:
            D1 = np.zeros((matrix.shape[0],), dtype=object)
            D2 = np.zeros((matrix.shape[0],), dtype=object)
            for i in range(matrix.shape[0]):
                D1[i] = np.sum([self.distance_1(wmatrix[i, j], NS[j]) for j in range(matrix.shape[1])])
                D2[i] = np.sum([self.distance_2(wmatrix[i, j], NS[j]) for j in range(matrix.shape[1])])
            D1_arr = np.array(D1, dtype=float)
            D2_arr = np.array(D2, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute distances to the negative solution: {e}") from e

        self._log_step(self.logger, 'D1 distances (Euclidean)', D1_arr, 'Euclidean distances from each alternative to NS')
        self._log_step(self.logger, 'D2 distances (Hamming)', D2_arr, 'Hamming distances from each alternative to NS')

        try:
            RA = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=object)
            for i in range(RA.shape[0]):
                for j in range(RA.shape[1]):
                    RA[i, j] = (D1[i] - D1[j]) + (_psi(D1[i] - D1[j], tau) * (D2[i] - D2[j]))
            RA_crisp = np.array(RA, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to build the Relative Assessment matrix: {e}") from e

        self._log_step(self.logger, 'Relative Assessment Matrix (RA)', RA_crisp, f'Pairwise comparisons using both distance measures (tau={tau})')

        try:
            AS = np.sum(RA, axis=1)
            AS_arr = np.array(AS, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute assessment scores: {e}") from e

        self._log_step(self.logger, 'Assessment Scores (AS)', AS_arr, 'Row sums of the relative assessment matrix — higher is better')

        return AS_arr