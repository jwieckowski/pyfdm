# Copyright (c) 2026 Jakub Więckowski

import numpy as np

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger
from ...validator import Validator

__all__ = ['fRANCOM']

class fRANCOM(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Ranking Comparison Method (F-RANCOM).

    Fuzzy RANCOM extends the crisp RANCOM method to a fuzzy environment. Either
    a criteria ranking is provided (from which a crisp Matrix of Ranking
    Comparison, MAC, is derived automatically via pairwise rank
    comparisons: 1.0 if criterion i outranks j, 0.5 if tied, 0.0
    otherwise), or a crisp MAC is provided directly. The crisp MAC is then
    fuzzified into a fuzzy MAC (fMAC) by mapping each of its three possible
    values (0.0, 0.5, 1.0) to a corresponding Triangular Fuzzy Number
    (TFN), and fuzzy weights are obtained by summing each criterion's row

    Parameters
    ----------
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.
    
    References
    ----------
    Więckowski, J., Kizielewicz, B., & Sałabun, W. (2025). Fuzzy RANCOM: 
    a novel approach for modeling uncertainty in decision-making processes. 
    Information sciences, 694, 121716.

    Examples
    --------
    >>> from pyfdm.weights.subjective import fRANCOM
    >>> francom = fRANCOM()
    >>> # criterion 0 most important, then 2, then 1
    >>> weights = francom(ranking=[1, 3, 2])
    """

    _TFN_MAP = {
        0.0: (0.0, 0.0, 0.5),
        0.5: (0.0, 0.5, 1.0),
        1.0: (0.5, 1.0, 1.0),
    }

    def __init__(self, logger: StepLogger | None = None):
        super().__init__(logger=logger)

    @staticmethod
    def _compare(rank_i: float, rank_j: float) -> float:
        """
        Pairwise comparison based on ranking positions.

        Parameters
        ----------
        rank_i : float
            Rank position of criterion i (lower means more important).
        rank_j : float
            Rank position of criterion j.

        Returns
        -------
        float
            ``1.0`` if criterion i outranks j (``rank_i < rank_j``),
            ``0.5`` if tied, ``0.0`` if criterion i is outranked by j.
        """

        if rank_i < rank_j:
            return 1.0

        if rank_i == rank_j:
            return 0.5

        return 0.0

    def _build_mac(self, ranking: np.ndarray) -> np.ndarray:
        """
        Build the crisp Matrix of Ranking Comparison (MAC) from a criteria
        ranking.

        Parameters
        ----------
        ranking : np.ndarray
            Rank position of each criterion, shape (n,). Lower values
            indicate higher importance; ties are allowed.

        Returns
        -------
        np.ndarray
            Crisp MAC of shape (n, n), with diagonal entries of 0.5 and
            ``mac[j, i] = 1 - mac[i, j]``.
        """

        n = len(ranking)

        mac = np.full((n, n), 0.5)

        for i in range(n):
            for j in range(i + 1, n):

                value = self._compare(ranking[i], ranking[j])

                mac[i, j] = value
                mac[j, i] = 1.0 - value

        return mac

    def _to_tfn(self, mac: np.ndarray) -> np.ndarray:
        """
        Convert a crisp MAC into the fuzzy MAC (fMAC) by mapping each entry
        to its corresponding TFN via `_TFN_MAP`.

        Parameters
        ----------
        mac : np.ndarray
            Crisp MAC of shape (n, n). Every entry must be exactly one of
            ``{0.0, 0.5, 1.0}``.

        Returns
        -------
        np.ndarray
            Fuzzy MAC of shape (n, n, 3), as (l, m, u) triples.

        Raises
        ------
        ValueError
            If `mac` contains any value not present in `_TFN_MAP` (i.e. not
            exactly 0.0, 0.5, or 1.0).
        """

        allowed = np.array(list(self._TFN_MAP.keys()))
        is_known = np.isin(mac, allowed)

        if not np.all(is_known):
            bad_values = np.unique(mac[~is_known])
            raise ValueError(
                f"'comparison_matrix' entries must be exactly one of "
                f"{sorted(self._TFN_MAP.keys())}, found unsupported "
                f"value(s): {bad_values.tolist()}."
            )

        n = mac.shape[0]

        fuzzy_mac = np.zeros((n, n, 3))

        for value, tfn in self._TFN_MAP.items():
            fuzzy_mac[mac == value] = tfn

        return fuzzy_mac

    def _calculate(
        self,
        ranking: np.ndarray | list | None = None,
        comparison_matrix: list[int | float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the F-RANCOM method.

        Either `ranking` must be provided (to derive a crisp MAC
        automatically) or `comparison_matrix` must be provided directly.
        If both are given, `comparison_matrix` takes precedence and
        `ranking` is only logged, not used to build the MAC.

        Parameters
        ----------
        ranking : np.ndarray | list | None, optional
            Rank position of each criterion, shape (n,). Lower values
            indicate higher importance; ties are allowed. Ignored (except
            for logging) if `comparison_matrix` is provided.
        comparison_matrix : list[int | float] | np.ndarray | None, optional
            Crisp Matrix of Ranking Comparison (MAC), shape (n, n). Every
            entry must be exactly one of ``{0.0, 0.5, 1.0}``.

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n, 3), as (l, m, u) triples.

        Raises
        ------
        ValueError
            If neither `ranking` nor `comparison_matrix` is provided, if
            `comparison_matrix` fails `Validator.validate_comparison_matrix`
            or contains unsupported values, or if `ranking`/
            `comparison_matrix` cannot be converted to float arrays.
        RuntimeError
            If the weight computation fails unexpectedly.
        """

        if comparison_matrix is None and ranking is None:
            raise ValueError("Either 'comparison_matrix' or 'ranking' must be provided.")

        if ranking is not None:
            try:
                ranking = np.asarray(ranking, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"'ranking' must be array-like numeric data convertible "
                    f"to a float numpy array: {e}"
                ) from e

            self._log_step(self.logger, "Criteria ranking", ranking)

        if comparison_matrix is not None:
            try:
                Validator.validate_comparison_matrix(comparison_matrix, dim=2)
            except Exception as e:
                raise ValueError(f"Invalid 'comparison_matrix': {e}") from e

            try:
                comparison_matrix = np.asarray(comparison_matrix, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"'comparison_matrix' must be array-like numeric data "
                    f"convertible to a float numpy array: {e}"
                ) from e

        mac = self._build_mac(ranking) if comparison_matrix is None else comparison_matrix

        self._log_step(self.logger, "Matrix of Ranking Comparison (MAC)", mac)

        fuzzy_mac = self._to_tfn(mac)

        self._log_step(self.logger, "Fuzzy MAC", fuzzy_mac)

        try:
            fuzzy_weights = np.sum(fuzzy_mac, axis=1)
            fuzzy_weights = fuzzy_weights / np.sum(fuzzy_weights, axis=0)[1]
        except Exception as e:
            raise RuntimeError(f"Failed to compute F-RANCOM weights: {e}") from e

        self._log_step(self.logger, "Fuzzy weights", fuzzy_weights)

        return fuzzy_weights