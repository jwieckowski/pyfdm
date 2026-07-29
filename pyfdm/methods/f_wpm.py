
# Copyright (c) 2023 - 2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..utils import mean_defuzzification
class fWPM(BaseFuzzyMethod):
    """
    Fuzzy Weighted Product Model (WPM).

    The method optionally normalizes the decision matrix and then raises
    each criterion value to the power of its corresponding criterion
    weight. The weighted criterion values are multiplied across all
    criteria to obtain a fuzzy preference value for each alternative,
    which is subsequently defuzzified into a crisp score. Higher scores
    indicate better alternatives.

    Parameters
    ----------
    defuzzify : callable, default=mean_defuzzification
        Function used to transform the final fuzzy preference values into
        crisp scores.
    normalization : callable, optional, default=None
        Function used to normalize the TFN decision matrix. If ``None``,
        the original decision matrix is used.
    logger : StepLogger | None, optional
        Optional logger used for recording intermediate computation steps.

    References
    ----------
    Triantaphyllou, E., & Lin, C. T. (1996). Development and evaluation
    of five fuzzy multiattribute decision-making methods. International
    Journal of Approximate reasoning, 14(4), 281-310.
    """

    def __init__(
        self,
        defuzzify: callable = mean_defuzzification,
        normalization: callable = None,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy WPM method object.

        Parameters
        ----------
        defuzzify : callable, default=mean_defuzzification
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

    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list | None = None,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy WPM method.

        This override exists to give `fWPM` its own call signature and
        docstring for IDEs/`help()`; all actual validation, computation,
        and logging happen in `BaseFuzzyMethod.__call__` / `fWPM._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,), optional, default = None
            Criteria types: ``1`` for benefit, ``-1`` for cost.

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy WPM preference score (total gap) for each
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
        >>> fwpm = fWPM()
        >>> scores = fwpm(matrix, weights, types)
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
        Weighted Product Model (WPM).

        The decision matrix is optionally normalized. Each criterion value
        is then raised to the power of its corresponding weight and the
        resulting weighted values are multiplied across all criteria.
        Finally, the fuzzy products are defuzzified to obtain crisp
        preference scores.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape ``(n,)``) or fuzzy
            (shape ``(n, 3)``). Crisp weights are automatically converted
            to triangular fuzzy weights.
        types : np.ndarray, shape (n,)
            Criteria types. Present for interface compatibility and ignored
            unless required by the selected normalization function.

        Returns
        -------
        np.ndarray, shape (m,)
            Weighted Product Model preference score for each alternative.
            Higher values indicate better alternatives.

        Raises
        ------
        RuntimeError
            If normalization, weighting, aggregation, or defuzzification
            fails unexpectedly.
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

        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            wmatrix = nmatrix ** weights
        except Exception as e:
            raise RuntimeError(f"Failed to apply weights using the weighted product model: {e}") from e

        self._log_step(
            self.logger,
            "Weighted matrix",
            np.array(wmatrix, dtype=float),
            "Each criterion value raised to the power of its corresponding weight"
        )

        try:
            prod_w = np.prod(wmatrix, axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate weighted criterion values: {e}") from e

        self._log_step(
            self.logger,
            "Weighted products",
            np.array(prod_w, dtype=float),
            "Product of weighted criterion values for each alternative"
        )

        try:
            result = np.array([self.defuzzify(s) for s in prod_w])
        except Exception as e:
            raise RuntimeError(f"Failed to defuzzify preference scores: {e}") from e

        self._log_step(
            self.logger,
            "Preference scores",
            result,
            "Defuzzified weighted products — higher is better"
        )

        return result

