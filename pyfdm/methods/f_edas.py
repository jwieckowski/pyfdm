# Copyright (c) 2022-2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import mean_defuzzification

class fEDAS(BaseFuzzyMethod):
    """
    Fuzzy Evaluation based on Distance from Average Solution (EDAS) method.

    The method computes the average fuzzy value for each criterion and
    evaluates every alternative according to its positive and negative
    distances from this average. After weighting and normalization, the
    appraisal score is obtained by averaging the normalized positive and
    negative distances. Higher scores indicate better alternatives.

    Parameters
    ----------
    defuzzify : callable
        Function used to transform Triangular Fuzzy Numbers into crisp
        values.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Zindani, D., Maity, S. R., & Bhowmik, S. (2019). Fuzzy-EDAS 
    (evaluation based on distance from average solution) for material 
    selection problems. In Advances in Computational Methods in 
    Manufacturing (pp. 755-771). Springer, Singapore.
    """

    def __init__(
        self,
        defuzzify: callable=mean_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy EDAS method object.

        Parameters
        ----------
        defuzzify : callable, default=mean_defuzzification
            Function used to defuzzify Triangular Fuzzy Numbers.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """
        super().__init__(logger=logger)
        self.defuzzify = defuzzify

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy EDAS method.

        This override exists to give `fEDAS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fEDAS._calculate`.

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
            fuzzy EDAS preference score for each alternative; higher is better.

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
        >>> types = np.array([1, 1, 1])
        >>> fedas = fEDAS()
        >>> scores = fedas(matrix, weights, types)
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
        EDAS method.

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
            Appraisal scores for all alternatives. Higher values indicate
            better alternatives.

        Raises
        ------
        RuntimeError
            If any stage of the EDAS computation fails.
        """

        def psi(a):
            return a if self.defuzzify(a) > 0 else [0, 0, 0]

        try:
            av_matrix = np.mean(matrix, axis=0)
            k = np.asarray([self.defuzzify(a) for a in av_matrix], dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute the average decision matrix: {e}") from e

        self._log_step(
            self.logger,
            "Average decision matrix (AV)",
            av_matrix,
            "Mean Triangular Fuzzy Number for each criterion."
        )

        try:
            pda = np.zeros(matrix.shape)
            nda = np.zeros(matrix.shape)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if types[j] == 1:
                        pda[i, j] = (
                            psi(matrix[i, j] - av_matrix[j][..., ::-1]) / k[j]
                        )
                        nda[i, j] = (
                            psi(av_matrix[j] - matrix[i, j][..., ::-1]) / k[j]
                        )
                    else:
                        pda[i, j] = (
                            psi(av_matrix[j] - matrix[i, j][..., ::-1]) / k[j]
                        )
                        nda[i, j] = (
                            psi(matrix[i, j] - av_matrix[j][..., ::-1]) / k[j]
                        )

        except Exception as e:
            raise RuntimeError(f"Failed to compute positive and negative distances from the average solution: {e}") from e

        self._log_step(
            self.logger,
            "Positive Distance from Average (PDA)",
            pda,
            "Positive distances of alternatives from the average solution."
        )

        self._log_step(
            self.logger,
            "Negative Distance from Average (NDA)",
            nda,
            "Negative distances of alternatives from the average solution."
        )

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            sp = np.sum(pda * weights, axis=1)
            sn = np.sum(nda * weights, axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to compute weighted positive and negative distances: {e}") from e

        self._log_step(
            self.logger,
            "Weighted positive distances (SP)",
            np.asarray(sp, dtype=float),
            "Weighted sum of positive distances."
        )

        self._log_step(
            self.logger,
            "Weighted negative distances (SN)",
            np.asarray(sn, dtype=float),
            "Weighted sum of negative distances."
        )

        try:
            nsp = sp / np.max([self.defuzzify(p) for p in sp])
            nsn = 1 - (sn / np.max([self.defuzzify(n) for n in sn]))
        except Exception as e:
            raise RuntimeError(f"Failed to normalize weighted distances: {e}") from e

        self._log_step(
            self.logger,
            "Normalized positive distances (NSP)",
            np.asarray(nsp, dtype=float),
            "Positive distances normalized by the maximum SP value."
        )

        self._log_step(
            self.logger,
            "Normalized negative distances (NSN)",
            np.asarray(nsn, dtype=float),
            "Negative distances transformed into benefit values."
        )

        try:
            appraisal = (nsp + nsn) / 2
            preferences = np.asarray([self.defuzzify(a) for a in appraisal], dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to compute final appraisal scores: {e}") from e

        self._log_step(
            self.logger,
            "Appraisal scores (AS)",
            preferences,
            "Final EDAS preference values. Higher values indicate better alternatives."
        )

        return preferences