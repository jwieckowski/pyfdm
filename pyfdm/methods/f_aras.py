# Copyright (c) 2022-2026 Jakub Więckowski

import numpy as np

from ._base import BaseFuzzyMethod
from ..validator import Validator
from ..step_logger import StepLogger
from ..utils.normalizations import sum_normalization

class fARAS(BaseFuzzyMethod):
    """
    Fuzzy Additive Ratio ASsessment (ARAS) method.

    Fuzzy ARAS ranks alternatives by comparing each one's overall utility
    (S_i) to that of a hypothetical optimal alternative (S_0), obtained by
    prepending an ideal row to the decision matrix (the per-criterion
    componentwise best TFN across alternatives), normalizing, and applying
    the weighted sum. The preference score is the ratio ``K_i = S_i / S_0``:
    the closer to 1, the closer the alternative is to the ideal.

    Parameters
    ----------
    normalization : callable, default=sum_normalization
        Function used to calculate the normalized decision matrix. Called
        as ``normalization(extended_matrix, types)``, where
        `extended_matrix` includes the prepended optimal-alternative row.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight 
    duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS,
    and MSGP methods. Mathematical Problems in Engineering, 2021.
    """

    def __init__(
        self,
        normalization=sum_normalization,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy ARAS method object.

        Parameters
        ----------
        normalization : callable, default=sum_normalization
            Function used to calculate the normalized decision matrix.
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
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Execute the fuzzy ARAS method.

        This override exists to give `fARAS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fARAS._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types (e.g. +1 for benefit, -1 for cost), forwarded to
            `self.normalization`.

        Returns
        -------
        np.ndarray, shape (m,)
            Preference score ``K_i = S_i / S_0`` for each alternative;
            higher is better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If the normalization step or the utility/score computation
            fails unexpectedly.

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
        >>> f_aras = fARAS()
        >>> scores = f_aras(matrix, weights, types)
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
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        ARAS method.

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

        Returns
        -------
        np.ndarray, shape (m,)
            Preference score ``K_i = S_i / S_0`` for each alternative;
            higher is better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If the normalization step or the utility/score computation
            fails unexpectedly.
        """

        try:
            exmatrix = np.zeros(
                (matrix.shape[0]+1, matrix.shape[1], matrix.shape[2]), dtype=object)
            exmatrix[1:] = matrix

            exmatrix[0, :] = np.repeat(
                np.max(np.max(matrix, axis=0), axis=1), 3
                    ).reshape(
                        matrix.shape[1], matrix.shape[2]
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to build the extended matrix: {e}") from e

        self._log_step(
            self.logger,
            'Extended matrix (with optimal row)',
            np.array(exmatrix, dtype=float),
            'Original matrix prepended with the optimal alternative row (componentwise max per criterion)'
        )

        try:
            nmatrix = self.normalization(exmatrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(
            self.logger,
            'Normalized matrix',
            np.array(nmatrix, dtype=float),
            'Result of applying normalization to the extended matrix'
        )

        try:
            if weights.ndim == 1:
                weights = np.repeat(weights, 3).reshape((len(weights), 3))

            wmatrix = nmatrix * weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights to the normalized matrix: {e}") from e

        self._log_step(
            self.logger,
            'Weighted normalized matrix',
            np.array(wmatrix, dtype=float),
            'Element-wise product of normalized matrix and weights'
        )

        try:
            S = 1 / 3 * np.sum(np.sum(wmatrix, axis=1), axis=1)
            S_arr = np.array(S, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute overall utility (S): {e}") from e

        self._log_step(
            self.logger,
            'Overall utility (S)',
            S_arr,
            'Sum of weighted normalized values — first entry is optimal alternative S0'
        )

        try:
            result = S_arr[1:] / S_arr[0]
        except Exception as e:
            raise RuntimeError(f"Failed to compute preference scores: {e}") from e

        self._log_step(
            self.logger,
            'Preference scores (K)',
            result,
            'Utility degree: K_i = S_i / S_0 — higher is better'
        )

        return result