# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import vector_normalization
from ..utils import vertex_distance

class fMAIRCA(BaseFuzzyMethod):
    """
    Fuzzy Multi-Attributive Ideal-Real Comparative Analysis (MAIRCA)
    method.

    The method compares, for every alternative and criterion, a
    Theoretical Ponder value (the a-priori chance of selecting the
    alternative, ``P = 1/m``, weighted by the criterion weight) against
    the corresponding Actual (Real) Ponder value obtained from the
    normalized decision matrix. The distance between the theoretical and
    real ponder matrices forms a gap matrix, and each alternative's total
    gap (summed across criteria) is its preference score. The SMALLER the
    total gap, the closer the alternative is to the theoretical ideal, so
    lower scores indicate better alternatives.

    Parameters
    ----------
    normalization : callable, default=vector_normalization
        Function used to normalize the TFN decision matrix.
    distance : callable, default=vertex_distance
        Function computing the distance between two individual TFNs.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.

    References
    ----------
    Boral, S., Howard, I., Chaturvedi, S. K., McKee, K., & Naikan, V. N. A. (2020).
    An integrated approach for fuzzy failure modes and effects analysis using
    fuzzy AHP and fuzzy MAIRCA. Engineering Failure Analysis, 108, 104195.
    """

    def __init__(
        self,
        normalization: callable = vector_normalization,
        distance: callable = vertex_distance,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy MAIRCA method object.

        Parameters
        ----------
        normalization : Callable, default=vector_normalization
            Function used to normalize the TFN decision matrix.
        distance : Callable, default=vertex_distance
            Function computing the distance between two individual TFNs.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
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
        Execute the fuzzy MAIRCA method.

        This override exists to give `fMAIRCA` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fMAIRCA._calculate`.

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
            fuzzy MAIRCA preference score (total gap) for each
            alternative; Lower is better (see `_descending`).

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
        >>> fmairca = fMAIRCA()
        >>> scores = fmairca(matrix, weights, types)
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
        MAIRCA method.

        The decision matrix is first normalized. A Theoretical Ponder
        matrix (TPA) is built from the uniform a-priori selection
        probability ``P = 1/m`` and the criterion weights, and an Actual
        (Real) Ponder matrix (TRA) is obtained by weighting the normalized
        matrix with TPA. The distance between TPA and TRA for every
        alternative-criterion pair forms a gap matrix, and each
        alternative's total gap (summed across criteria) is its final
        preference score.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape ``(n,)``) or fuzzy
            (shape ``(n, 3)``). Crisp weights are automatically converted
            to triangular fuzzy weights.
        types : np.ndarray, shape (n,)
            Criteria types (``1`` for benefit criteria, ``-1`` for cost
            criteria), forwarded to the selected normalization function.

        Returns
        -------
        np.ndarray, shape (m,)
            Total gap score for each alternative. LOWER values indicate
            better alternatives (closer to the theoretical ideal).

        Raises
        ------
        RuntimeError
            If normalization, ponder matrix computation, distance
            calculation, or aggregation fails unexpectedly.
        """

        P = 1 / matrix.shape[0]

        try:
            tpa = np.ones(matrix.shape, dtype=object)
            for j in range(matrix.shape[1]):
                tpa[:, j] = P * weights[j]
        except Exception as e:
            raise RuntimeError(f"Failed to build the Theoretical Ponder matrix (TPA): {e}") from e

        self._log_step(self.logger, 'Theoretical Ponder Matrix (TPA)', np.array(tpa, dtype=float), f'Uniform selection probability P=1/{matrix.shape[0]} multiplied by weights')

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(self.logger, 'Normalized matrix', nmatrix, 'Result of applying normalization function')

        try:
            tra = nmatrix * tpa
        except Exception as e:
            raise RuntimeError(f"Failed to build the Actual (Real) Ponder matrix (TRA): {e}") from e

        self._log_step(self.logger, 'Actual Ponder Matrix (TRA)', np.array(tra, dtype=float), 'Element-wise product of normalized matrix and TPA')

        try:
            d = np.zeros((matrix.shape[0], matrix.shape[1], 1))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    d[i, j] = self.distance(tpa[i, j], tra[i, j])
            d_arr = d[:, :, 0]
        except Exception as e:
            raise RuntimeError(f"Failed to compute the gap (distance) matrix: {e}") from e

        self._log_step(self.logger, 'Gap matrix (D)', d_arr, 'Distance between TPA and TRA for each alternative-criterion pair')

        try:
            Q = np.sum(d, axis=1).flatten()
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate the total gap scores: {e}") from e

        self._log_step(self.logger, 'Preference scores (Q)', np.array(Q, dtype=float), 'Row sums of the gap matrix — lower is better')
        return Q