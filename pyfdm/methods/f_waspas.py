# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import waspas_normalization, mean_defuzzification

class fWASPAS(BaseFuzzyMethod):
    """
    Fuzzy Weighted Aggregated Sum Product Assessment (WASPAS)
    method.

    The method combines the additive Weighted Sum Model (WSM) and the
    multiplicative Weighted Product Model (WPM). After normalizing the
    fuzzy decision matrix, two weighted matrices are constructed:
    one for the additive component and one for the multiplicative
    component. Their defuzzified utility values are then combined using
    an adaptive coefficient d = ΣP / (ΣQ + ΣP), where Q denotes the 
    WSM utility values and P the WPM utility values.
    Higher scores indicate better alternatives.

    .. rubric:: Reference

        Turskis, Z., Zavadskas, E. K., Antuchevičienė, J., & Kosareva, N.
        (2015). A hybrid model based on fuzzy AHP and fuzzy WASPAS for 
        construction site selection.

    Parameters
    ----------
    normalization : callable, default=waspas_normalization
        Function used to normalize the TFN decision matrix.
    defuzzify : callable, default=mean_defuzzification
        Function used to convert triangular fuzzy numbers into crisp
        values.
    logger : StepLogger | None, optional
        Optional logger used for recording intermediate computation steps.
    """

    def __init__(
        self,
        normalization: callable = waspas_normalization,
        defuzzify: callable = mean_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy WASPAS method object.

        Parameters
        ----------
        normalization : callable, default=waspas_normalization
            Function used to normalize the TFN decision matrix.
        defuzzify : callable, default=mean_defuzzification
            Function used to convert TFNs into crisp values.
        logger : StepLogger | None, optional
            Optional logger used for recording computation steps.
        """
        super().__init__(logger=logger)
        self.normalization = normalization
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
        Execute the fuzzy WASPAS method.

        This override exists to give `fWASPAS` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fWASPAS._calculate`.

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
            fuzzy WASPAS preference score (total gap) for each
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
        >>> fwaspas = fWPASPAS()
        >>> scores = fwaspas(matrix, weights, types)
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
        WASPAS method.

        The decision matrix is first normalized and then evaluated using
        both the Weighted Sum Model (WSM) and the Weighted Product Model
        (WPM). Their defuzzified utility values are combined through an
        adaptive aggregation coefficient to produce the final WASPAS score.

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
            criteria).

        Returns
        -------
        np.ndarray, shape (m,)
            Final WASPAS preference score for each alternative. Higher
            values indicate better alternatives.

        Raises
        ------
        RuntimeError
            If normalization, weighting, utility computation,
            defuzzification, adaptive coefficient calculation,
            or final aggregation fails unexpectedly.
        """

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(
            self.logger,
            'Normalized matrix',
            nmatrix,
            'Result of applying normalization function'
        )

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wsm_wmatrix = nmatrix * weights
            wpm_wmatrix = nmatrix ** weights[..., ::-1]
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct weighted matrices: {e}"
            ) from e

        self._log_step(
            self.logger,
            'WSM weighted matrix',
            wsm_wmatrix,
            'Normalized matrix multiplied by criterion weights'
        )

        self._log_step(
            self.logger,
            'WPM weighted matrix',
            wpm_wmatrix,
            'Normalized matrix raised to fuzzy criterion weights'
        )

        try:
            Q = np.sum(wsm_wmatrix, axis=1)
            P = np.prod(wpm_wmatrix, axis=1)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute WSM/WPM utility values: {e}"
            ) from e

        try:
            Q_def = np.array([self.defuzzify(q) for q in Q])
            P_def = np.array([self.defuzzify(p) for p in P])
        except Exception as e:
            raise RuntimeError(
                f"Failed to defuzzify utility values: {e}"
            ) from e

        self._log_step(
            self.logger,
            'WSM scores (Q)',
            Q_def,
            'Defuzzified additive utility values'
        )

        self._log_step(
            self.logger,
            'WPM scores (P)',
            P_def,
            'Defuzzified multiplicative utility values'
        )

        try:
            d = np.sum(P_def) / (np.sum(Q_def) + np.sum(P_def))
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute adaptive coefficient d: {e}"
            ) from e

        self._log_step(
            self.logger,
            'Adaptive coefficient (d)',
            np.array([d]),
            'Relative contribution of the WPM component'
        )

        try:
            result = d * Q_def + (1 - d) * P_def
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute final WASPAS scores: {e}"
            ) from e

        self._log_step(
            self.logger,
            'Preference scores',
            result,
            'Final WASPAS utility values — higher is better'
        )

        return result
