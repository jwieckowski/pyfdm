# Copyright (c) 2026 Jakub Więckowski

import numpy as np

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger
from ...validator import Validator

__all__ = ['fLMAW']

class fLMAW(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Logarithm Methodology of Additive Weights (F-LMAW).

    Fuzzy LMAW derives criteria weights from expert-assigned significance
    ratings (linguistic terms or raw TFNs) on a bounded scale, relative to
    an anti-ideal point (the smallest possible rating on that scale). Each
    rating is related to the anti-ideal point via fuzzy division
    (``mu_j = gamma_j / gamma_A``), and weights are obtained via a
    logarithmic normalization (``w_j = ln(mu_j) / ln(prod_k mu_k)``) which
    guarantees that, for a single expert, the resulting weights sum to 1.
    When multiple experts are provided, individual expert weights are
    aggregated criterion-wise via the geometric mean.

    Attributes
    ----------
    gamma_a : np.ndarray
        The anti-ideal point as a float array of shape (3,).
    linguistic_scale : dict[str, tuple[float, float, float]]
        The linguistic scale in effect (either the one passed in, or
        `DEFAULT_SCALE`).

    Parameters
    ----------
    anti_ideal_point : list[float] | np.ndarray, default=(0.5, 0.5, 0.5)
        Anti-ideal TFN ``(l, m, u)``, representing the lowest possible
        significance rating on the scale in use. Must satisfy
        ``l <= m <= u``.
    linguistic_scale : dict[str, tuple[float, float, float]] | None, optional
        Mapping from linguistic terms to TFNs, used to interpret string
        entries in `expert_decisions`. If None (default), `DEFAULT_SCALE`
        (a 9-point scale from 'AL' to 'AH') is used.
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.
    
    References
    ----------
    Božanić, D., Pamučar, D., Milić, A., Marinković, D., & Komazec, N. (2022). 
    Modification of the logarithm methodology of additive weights (LMAW) 
    by a triangular fuzzy number and its application in multi-criteria 
    decision making. Axioms, 11(3), 89.

    Examples
    --------
    >>> from pyfdm.weights.subjective import fLMAW
    >>> # single expert, linguistic ratings for 3 criteria
    >>> flmaw = fLMAW()
    >>> weights = flmaw(expert_decisions=[['H', 'MH', 'VH']])
    """

    DEFAULT_SCALE = {
        "AH": (4.5, 5.0, 5.0),
        "VH": (4.0, 4.5, 5.0),
        "H":  (3.5, 4.0, 4.5),
        "MH": (3.0, 3.5, 4.0),
        "E":  (2.5, 3.0, 3.5),
        "ML": (2.0, 2.5, 3.0),
        "L":  (1.5, 2.0, 2.5),
        "VL": (1.0, 1.5, 2.0),
        "AL": (1.0, 1.0, 1.0),
    }

    def __init__(
        self,
        anti_ideal_point: list[float] | np.ndarray = (0.5, 0.5, 0.5),
        linguistic_scale: dict[str, tuple[float, float, float]] | None = None,
        logger: StepLogger | None = None,
    ):

        super().__init__(logger=logger)

        try:
            gamma_a = np.asarray(anti_ideal_point, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'anti_ideal_point' must be array-like numeric data "
                f"convertible to a float numpy array: {e}"
            ) from e

        if gamma_a.shape != (3,):
            raise ValueError(
                f"'anti_ideal_point' must be a single TFN (l, m, u), "
                f"got shape {gamma_a.shape}."
            )

        if not (gamma_a[0] <= gamma_a[1] <= gamma_a[2]):
            raise ValueError(f"'anti_ideal_point' must satisfy l <= m <= u, got {tuple(gamma_a)}.")

        self.gamma_a = gamma_a

        self.linguistic_scale = (
            linguistic_scale
            if linguistic_scale is not None
            else self.DEFAULT_SCALE
        )

    def _prepare_gamma(
        self,
        expert_decisions: list[list[str]] | list[list[list[float]]] | np.ndarray
    ) -> np.ndarray:
        """
        Normalizes expert significance ratings into a fuzzy array.

        Accepts either linguistic terms (resolved via `self.linguistic_scale`)
        or raw TFN triples, for a single expert or multiple experts.

        Parameters
        ----------
        expert_decisions : list[list[str]] | list[list[list[float]]] | np.ndarray
            Expert significance ratings, one of:
            - linguistic terms, shape (n_criteria,) for a single expert or
              (n_experts, n_criteria) for multiple experts;
            - raw TFNs, shape (n_criteria, 3) for a single expert or
              (n_experts, n_criteria, 3) for multiple experts.

        Returns
        -------
        np.ndarray
            Fuzzy ratings of shape (n_experts, n_criteria, 3).

        Raises
        ------
        ValueError
            If `expert_decisions` fails `Validator.validate_expert_decisions`,
            has an unsupported shape/number of dimensions, or contains
            unknown linguistic terms.
        """

        try:
            Validator.validate_lmaw_input(
                expert_decisions,
                linguistic_scale=self.linguistic_scale,
            )
        except Exception as e:
            raise ValueError(f"Invalid 'expert_decisions': {e}") from e

        arr = np.asarray(expert_decisions, dtype=object)

        if arr.size == 0:
            raise ValueError("'expert_decisions' must not be empty.")

        first = arr.flat[0]

        if not isinstance(first, str):
            # raw TFNs
            if arr.ndim == 2:
                if arr.shape[-1] != 3:
                    raise ValueError(
                        f"Numeric 'expert_decisions' with 2 dimensions must "
                        f"have shape (n_criteria, 3), got {arr.shape}."
                    )
                return arr.astype(float)[None, :, :]

            if arr.ndim == 3:
                if arr.shape[-1] != 3:
                    raise ValueError(
                        f"Numeric 'expert_decisions' with 3 dimensions must "
                        f"have shape (n_experts, n_criteria, 3), got {arr.shape}."
                    )
                return arr.astype(float)

            raise ValueError(
                f"Numeric 'expert_decisions' must have 2 dimensions "
                f"(n_criteria, 3) or 3 dimensions (n_experts, n_criteria, 3), "
                f"got {arr.ndim} dimensions with shape {arr.shape}."
            )

        # linguistic terms
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Linguistic 'expert_decisions' must have 1 dimension "
                f"(n_criteria,) or 2 dimensions (n_experts, n_criteria), "
                f"got {arr.ndim} dimensions with shape {arr.shape}."
            )

        try:
            mapped = [
                [self.linguistic_scale[token] for token in expert]
                for expert in arr
            ]
        except KeyError as e:
            raise ValueError(
                f"Unknown linguistic term {e}. "
                f"Available: {list(self.linguistic_scale.keys())}"
            ) from e

        return np.asarray(mapped, dtype=float)

    def _calculate(
        self,
        expert_decisions: list[list[str]] | list[list[list[float]]] | np.ndarray
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the F-LMAW method.

        Parameters
        ----------
        expert_decisions : list[list[str]] | list[list[list[float]]] | np.ndarray
            Expert significance ratings; see `_prepare_gamma` for accepted
            shapes/formats.

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n_criteria, 3), as (l, m, u) triples,
            aggregated across experts via the geometric mean.

        Raises
        ------
        ValueError
            If `expert_decisions` is invalid (propagated from
            `_prepare_gamma`).
        RuntimeError
            If the weight computation fails unexpectedly (e.g. due to
            non-finite intermediate values).
        """

        gamma = self._prepare_gamma(expert_decisions)
        self._log_step(self.logger, "Gamma", gamma)

        n_experts, n_criteria, _ = gamma.shape

        try:
            # Fuzzy relation: mu = gamma / gamma_a
            mu = np.zeros_like(gamma)

            mu[:, :, 0] = gamma[:, :, 0] / self.gamma_a[2]
            mu[:, :, 1] = gamma[:, :, 1] / self.gamma_a[1]
            mu[:, :, 2] = gamma[:, :, 2] / self.gamma_a[0]

            self._log_step(self.logger, "Fuzzy relation (mu)", mu)

            # Weight coefficients
            coeff = np.zeros_like(mu)

            for e in range(n_experts):

                prod_l = np.prod(mu[e, :, 0])
                prod_m = np.prod(mu[e, :, 1])
                prod_u = np.prod(mu[e, :, 2])

                for j in range(n_criteria):

                    if abs(prod_u - 1.0) > 1e-12:
                        coeff[e, j, 0] = (
                            np.log(mu[e, j, 0])
                            / np.log(prod_u)
                        )

                    if abs(prod_m - 1.0) > 1e-12:
                        coeff[e, j, 1] = (
                            np.log(mu[e, j, 1])
                            / np.log(prod_m)
                        )

                    if abs(prod_l - 1.0) > 1e-12:
                        coeff[e, j, 2] = (
                            np.log(mu[e, j, 2])
                            / np.log(prod_l)
                        )

            self._log_step(self.logger, "Weight coefficients", coeff)

            # Aggregate judgments across experts (geometric mean)
            fuzzy_weights = np.zeros((n_criteria, 3))

            for j in range(n_criteria):

                fuzzy_weights[j, 0] = (
                    np.prod(coeff[:, j, 0])
                    ** (1 / n_experts)
                )

                fuzzy_weights[j, 1] = (
                    np.prod(coeff[:, j, 1])
                    ** (1 / n_experts)
                )

                fuzzy_weights[j, 2] = (
                    np.prod(coeff[:, j, 2])
                    ** (1 / n_experts)
                )
        except Exception as e:
            raise RuntimeError(f"Failed to compute F-LMAW weights: {e}") from e

        self._log_step(self.logger, "Fuzzy weights", fuzzy_weights)

        return fuzzy_weights