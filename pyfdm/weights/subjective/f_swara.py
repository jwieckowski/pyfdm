# Copyright (c) 2026 Jakub Więckowski

import numpy as np

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger

__all__ = ['fSWARA']

class fSWARA(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Step-wise Weight Assessment Ratio Analysis (F-SWARA).

    Criteria are ranked from most to least important, and for every
    criterion after the first, an expert states its comparative importance
    (as a Triangular Fuzzy Number, TFN) relative to the previous one in the
    ranking. This significance S_j is turned into a coefficient
    ``K_j = S_j + 1``, and unnormalized weight coefficients q_j are derived
    recursively (``q_1 = 1``, ``q_j = q_{j-1} / K_j``) using proper fuzzy
    (TFN) division, then normalized (again via fuzzy division) so that the
    modal (m) components sum to 1. When multiple experts provide
    comparative importance judgments, they are aggregated (weighted mean,
    or plain mean if `expert_weights` is None) before computing q_j.

    .. rubric:: Reference
        
        Mehdiabadi, A., Sadeghi, A., Karbassi Yazdi, A., & Tan, Y. (2025).
        Sustainability Service Chain Capabilities in the Oil and Gas Industry:
        A Fuzzy Hybrid Approach SWARA-MABAC.
        Spectrum of Operational Research, 2(1), 114-134.

    Parameters
    ----------
    scale : dict[str, tuple[float, float, float]] | None, optional
        Mapping from linguistic terms to TFNs, used to interpret string
        entries in `comparative_importance`. If None (default),
        `DEFAULT_SCALE` (a 5-point scale from 'EI' to 'MI') is used.
    expert_weights : list[float] | np.ndarray | None, optional
        Weights used to aggregate multiple experts' comparative importance
        judgments via a weighted average. Must have length equal to the
        number of experts and sum to a positive value. If None (default), a
        plain (unweighted) mean is used.
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.

    Attributes
    ----------
    scale : dict[str, tuple[float, float, float]]
        The linguistic scale in effect (either the one passed in, or
        `DEFAULT_SCALE`).
    expert_weights : list[float] | np.ndarray | None
        The expert weights in effect.

    Examples
    --------
    >>> from pyfdm.weights.subjective import fSWARA
    >>> # criterion 2 most important, then 1, then 3
    >>> ranking = [2, 1, 3]
    >>> # comparative importance of rank-2 vs rank-1, and rank-3 vs rank-2
    >>> comparative_importance = ['MLI', 'LI']
    >>> fswara = fSWARA()
    >>> weights = fswara(ranking=ranking, comparative_importance=comparative_importance)
    """

    DEFAULT_SCALE = {
        "EI":  (1.0, 1.0, 1.0),          # Equally important
        "MLI": (2/3, 1.0, 3/2),          # Moderately less important
        "LI":  (3/2, 2.0, 5/2),          # Less important
        "VLI": (5/2, 3.0, 7/2),          # Very less important
        "MI":  (7/2, 4.0, 9/2),          # Much less important
    }

    def __init__(
        self,
        scale: dict[str, tuple[float, float, float]] | None = None,
        expert_weights: list[float] | np.ndarray | None = None,
        logger: StepLogger | None = None,
    ):

        super().__init__(logger=logger)
        self.scale = (
            scale
            if scale is not None
            else self.DEFAULT_SCALE
        )

        if expert_weights is not None:
            try:
                expert_weights = np.asarray(expert_weights, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"'expert_weights' must be array-like numeric data "
                    f"convertible to a float numpy array: {e}"
                ) from e

            if expert_weights.ndim != 1 or expert_weights.size == 0:
                raise ValueError(
                    "'expert_weights' must be a non-empty 1D array of weights."
                )

            if np.any(expert_weights < 0) or np.sum(expert_weights) <= 0:
                raise ValueError(
                    "'expert_weights' must be non-negative and sum to a "
                    "positive value."
                )

        self.expert_weights = expert_weights

    @staticmethod
    def _tfn_divide(
        a: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        Approximate TFN division: ``a / b ≈ (a_l/b_l, a_m/b_m, a_u/b_u)``.

        Parameters
        ----------
        a : np.ndarray
            Dividend TFN(s), shape (..., 3).
        b : np.ndarray
            Divisor TFN(s), shape (..., 3), broadcastable with `a`.

        Returns
        -------
        np.ndarray
            The resulting TFN(s), same shape as the broadcast of `a`/`b`.
        """

        return np.stack(
            [
                a[..., 0] / b[..., 0],
                a[..., 1] / b[..., 1],
                a[..., 2] / b[..., 2],
            ],
            axis=-1
        )

    def _prepare_gamma(
        self,
        comparative_importance: list[str] | list[list[str]] | list[list[float]] | np.ndarray
    ) -> np.ndarray:
        """
        Normalizes expert comparative-importance judgments into a fuzzy
        array.

        Accepts either linguistic terms (resolved via `self.scale`) or raw
        TFN triples, for a single expert or multiple experts.

        Parameters
        ----------
        comparative_importance : list[str] | list[list[str]] | list[list[float]] | np.ndarray
            Comparative importance judgments, one of:
            - linguistic terms, shape (n_criteria-1,) for a single expert or
              (n_experts, n_criteria-1) for multiple experts;
            - raw TFNs, shape (n_criteria-1, 3) for a single expert or
              (n_experts, n_criteria-1, 3) for multiple experts.

        Returns
        -------
        np.ndarray
            Fuzzy judgments of shape (n_experts, n_criteria-1, 3).

        Raises
        ------
        ValueError
            If `comparative_importance` has an unsupported shape/number of
            dimensions, or contains unknown linguistic terms.
        """

        arr = np.asarray(comparative_importance)

        # TFN input
        if np.issubdtype(arr.dtype, np.number):

            if arr.ndim == 2 and arr.shape[-1] == 3:
                # one expert
                return arr.astype(float)[None, :, :]

            if arr.ndim == 3 and arr.shape[-1] == 3:
                # multiple experts
                return arr.astype(float)

            raise ValueError(
                f"Numeric 'comparative_importance' must have shape "
                f"(n_criteria-1, 3) or (n_experts, n_criteria-1, 3), got "
                f"{arr.shape}."
            )

        # Linguistic input
        arr = np.asarray(comparative_importance, dtype=object)

        if arr.ndim == 1:
            # one expert
            arr = arr.reshape(1, -1)

        if arr.ndim == 2:
            try:
                gamma = [[self.scale[value] for value in expert] for expert in arr]
                return np.asarray(gamma, dtype=float)

            except KeyError as e:
                raise ValueError(
                    f"Unknown linguistic value: {e.args[0]}. "
                    f"Available values: {list(self.scale.keys())}"
                ) from e

        raise ValueError("Invalid comparative_importance format.")

    def _calculate(
        self,
        ranking: np.ndarray | list,
        comparative_importance: np.ndarray | list,
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the F-SWARA method.

        Parameters
        ----------
        ranking : np.ndarray | list
            Ordered, 1-based indices of criteria, most important first
            (i.e. a permutation of ``1..n``).
        comparative_importance : np.ndarray | list
            Comparative importance of each criterion (after the first)
            relative to the previous one in the ranking; see
            `_prepare_gamma` for accepted shapes/formats. Must resolve to
            exactly ``n_criteria - 1`` judgments per expert.

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n_criteria, 3), as (l, m, u) triples,
            indexed by ORIGINAL criterion index (not by rank).

        Raises
        ------
        ValueError
            If `ranking` is not a valid permutation of ``1..n``, if
            `comparative_importance` does not resolve to exactly
            ``n_criteria - 1`` judgments per expert, or if it fails
            `_prepare_gamma`.
        RuntimeError
            If the weight computation fails unexpectedly.
        """

        try:
            ranking = np.asarray(ranking, dtype=int)
        except (TypeError, ValueError) as e:
            raise ValueError(f"'ranking' must be array-like integer data: {e}") from e

        n_criteria = len(ranking)

        if sorted(ranking.tolist()) != list(range(1, n_criteria + 1)):
            raise ValueError(
                "'ranking' must be a permutation of 1..n (1-based criterion "
                f"indices), got {ranking.tolist()}."
            )

        gamma = self._prepare_gamma(comparative_importance)

        if gamma.shape[1] != n_criteria - 1:
            raise ValueError(
                f"'comparative_importance' must provide exactly "
                f"{n_criteria - 1} judgments per expert (n_criteria - 1), "
                f"got {gamma.shape[1]}."
            )

        if self.expert_weights is not None and len(self.expert_weights) != gamma.shape[0]:
            raise ValueError(
                f"'expert_weights' length ({len(self.expert_weights)}) must "
                f"match the number of experts in 'comparative_importance' "
                f"({gamma.shape[0]})."
            )

        self._log_step(self.logger, "Comparative importance (TFN)", gamma)

        try:
            # Aggregate experts
            if self.expert_weights is None:
                Sj = np.mean(gamma, axis=0)
            else:
                Sj = np.average(gamma, axis=0, weights=self.expert_weights)

            self._log_step(self.logger, "Aggregated S_j", Sj)

            Kj = Sj + 1.0
            self._log_step(self.logger, "K_j", Kj)

            q = np.ones((n_criteria, 3), dtype=float)
            for i in range(1, n_criteria):
                q[i] = self._tfn_divide(q[i - 1], Kj[i - 1])

            self._log_step(self.logger, "q_j", q)

            q_sum = np.sum(q, axis=0)
            fuzzy_weights = self._tfn_divide(q, q_sum)
        except Exception as e:
            raise RuntimeError(f"Failed to compute F-SWARA weights: {e}") from e

        self._log_step(self.logger, "Fuzzy weights", fuzzy_weights)

        # restore original criterion order
        result = np.zeros_like(fuzzy_weights)
        for rank, criterion in enumerate(ranking):
            result[criterion - 1] = fuzzy_weights[rank]

        self._log_step(self.logger, "Weights in original criterion order", result)

        return result