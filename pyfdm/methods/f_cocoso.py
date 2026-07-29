# Copyright (c) 2023-2026 Jakub Więckowski

import numpy as np
from typing import Any

from ._base import BaseFuzzyMethod
from ..step_logger import StepLogger
from ..validator import Validator

from ..utils.normalizations import cocoso_normalization
from ..utils.defuzzifications import mean_defuzzification

class fCOCOSO(BaseFuzzyMethod):
    """
    Fuzzy Combined Compromise Solution (COCOSO) method.

    Normalizes the TFN decision matrix, computes a weighted-sum component
    (S, "WSM") and a weighted-power component (P, "WPM") per alternative,
    combines them via three complementary fuzzy assessment strategies
    (fa: balanced, fb: relative performance, fc: compromise with parameter
    `d`), defuzzifies each strategy's scores, and aggregates them into a
    single final score via the geometric-mean/arithmetic-mean combination
    of Yazdani et al.'s COCOSO. Higher scores indicate better alternatives.

    Parameters
    ----------
    normalization : callable
        Function used to normalize the TFN decision matrix.
    defuzzify : callable
        Function used to defuzzify a TFN into a crisp value.
    d : float
        Compromise parameter blending S and P in strategy `fc`.
    logger : StepLogger | None, optional
        Optional logger used for recording computation steps.
    
    References
    ----------
    Ulutaş, A., Popovic, G., Radanov, P., Stanujkic, D., & Karabasevic, D. (2021).
    A new hybrid fuzzy PSI-PIPRECIA-CoCoSo MCDM based approach to solving the 
    transportation company selection problem. Technological and Economic 
    Development of Economy, 27(5), 1227-1249.
    """

    def __init__(
        self,
        normalization=cocoso_normalization,
        defuzzify=mean_defuzzification,
        logger: StepLogger | None = None
    ):
        """
        Create a fuzzy COCOSO method object.

        Parameters
        ----------
        normalization : callable, default=cocoso_normalization
            Function used to normalize the TFN decision matrix.
        defuzzify : callable, default=mean_defuzzification
            Function used to defuzzify a TFN into a crisp value.
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
        d: float = 0.5,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute the fuzzy COCOSO method.

        This override exists purely to surface COCOSO's specific tuning
        parameters (`d`) in the call signature/docstring for IDEs and `help()`; 
        all actual validation, computation, and logging happen in 
        `BaseFuzzyMethod.__call__` / `fCOCOSO._calculate`.

        Parameters
        ----------
        matrix : np.ndarray | list, shape (m, n, 3)
            TFN decision matrix.
        weights : np.ndarray | list, shape (n,) or (n, 3)
            Criterion weights, either crisp (shape (n,)) or fuzzy
            (shape (n, 3)).
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        d : float, default=0.5
            Compromise parameter blending S and P in strategy `fc`
            (weight `d` for S, `1 - d` for P). Determined by the
            decision-maker.

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy COCOSO preference score for each alternative; higher is better.

        Raises
        ------
        ValueError
            If input shapes are inconsistent, or `d` fail validation (see `_calculate`).
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
        >>> fcocoso = fCOCOSO()
        >>> scores = fcocoso(matrix, weights, types, d=0.7)
        """
        return super().__call__(
            matrix,
            weights,
            types,
            d=d,
            *args,
            **kwargs
        )

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        d: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform validation specific for the fuzzy COCOSO inputs.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.
        weights : np.ndarray | list
            Criterion weights.
        types : np.ndarray | list, shape (n,)
            Criteria types: ``1`` for benefit, ``-1`` for cost.
        d : float, default=0.5
            Compromise parameter blending S and P in strategy `fc`
            (weight `d` for S, `1 - d` for P). Determined by the
            decision-maker.

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

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate fCOCOSO input: {e}") from e

    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        d: float,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Computes preference scores for each alternative using the fuzzy
        COCOSO method.

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
        d : float,
            Compromise parameter blending S and P in strategy `fc`
            (weight `d` for S, `1 - d` for P). Determined by the
            decision-maker.

        Returns
        -------
        np.ndarray, shape (m,)
            fuzzy COCOSO preference score for each alternative; higher is
            better.

        Raises
        ------
        ValueError
            If `matrix`/`weights`/`types` have inconsistent shapes.
        RuntimeError
            If normalization, defuzzification, or any aggregation step
            fails unexpectedly (e.g. due to non-finite intermediate
            values).
        """

        try:
            nmatrix = self.normalization(matrix, types)
        except Exception as e:
            raise RuntimeError(f"Normalization step failed: {e}") from e

        self._log_step(
            self.logger,
            'Normalized matrix',
            np.array(nmatrix, dtype=float),
            'Result of applying normalization function'
        )

        # Convert crisp weights into TFNs
        if weights.ndim == 1:
            weights = np.repeat(weights, 3).reshape((len(weights), 3))

        try:
            # Weighted sum and weighted product components
            S = np.sum(nmatrix * weights, axis=1)
            P = np.sum(nmatrix ** weights[..., ::-1], axis=1)
        except Exception as e:
            raise RuntimeError(f"Failed to compute S/P components: {e}") from e

        self._log_step(
            self.logger,
            'Sum of comparability (S)',
            np.array(S, dtype=float),
            'Weighted sum of normalised values (WSM component)'
        )
        self._log_step(
            self.logger,
            'Sum of power weights (P)',
            np.array(P, dtype=float),
            'Weighted product of normalised values (WPM component)'
        )

        try:
            # Three fuzzy assessment strategies
            fa = np.array([
                (P[i, :] + S[i, :]) /
                (np.sum(P + S, axis=0)[..., ::-1])
                for i in range(matrix.shape[0])
            ])

            fb = np.array([
                S[i, :] / np.min(S) +
                P[i, :] / np.min(P)
                for i in range(matrix.shape[0])
            ])

            fc = np.array([
                (d * S[i, :] + (1 - d) * P[i, :]) /
                (d * np.max(S) + (1 - d) * np.max(P))
                for i in range(matrix.shape[0])
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to compute fuzzy assessment strategies: {e}") from e

        self._log_step(
            self.logger,
            'Fuzzy score fa',
            np.array(fa, dtype=float),
            '(P + S) / total(P + S) — balanced strategy'
        )
        self._log_step(
            self.logger,
            'Fuzzy score fb',
            np.array(fb, dtype=float),
            'S/min(S) + P/min(P) — relative performance strategy'
        )
        self._log_step(
            self.logger,
            'Fuzzy score fc',
            np.array(fc, dtype=float),
            f'Compromise strategy with d={d}: '
            'd·S + (1-d)·P normalization'
        )

        try:
            # Defuzzification
            nfa = np.array([self.defuzzify(f) for f in fa])
            nfb = np.array([self.defuzzify(f) for f in fb])
            nfc = np.array([self.defuzzify(f) for f in fc])
        except Exception as e:
            raise RuntimeError(f"Defuzzification failed: {e}") from e

        self._log_step(
            self.logger,
            'Crisp fa',
            nfa,
            'Defuzzified balanced strategy scores'
        )
        self._log_step(
            self.logger,
            'Crisp fb',
            nfb,
            'Defuzzified relative strategy scores'
        )
        self._log_step(
            self.logger,
            'Crisp fc',
            nfc,
            'Defuzzified compromise strategy scores'
        )

        try:
            # Final COCOSO aggregation
            result = (
                (nfa * nfb * nfc) ** (1 / 3)
                +
                (nfa + nfb + nfc) / 3
            ) / 2
        except Exception as e:
            raise RuntimeError(f"Failed to compute final assessment score: {e}") from e

        self._log_step(
            self.logger,
            'Final assessment score',
            result,
            'COCOSO aggregated preference score '
            '(geometric and arithmetic mean of fa, fb, fc)'
        )

        return result