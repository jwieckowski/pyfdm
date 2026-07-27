# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from scipy.optimize import linprog

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger

__all__ = ['fFUCOM']

class fFUCOM(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Full Consistency Method (FUCOM-F).

    FUCOM-F extends the crisp Full Consistency Method to a fuzzy
    environment: criteria are ranked from most to least important, and each
    ranked criterion is assigned a fuzzy significance (as a Triangular
    Fuzzy Number, TFN) relative to the top-ranked one, using a linguistic
    scale ('EI', 'WI', 'FI', 'VI', 'AI' by default) or raw (l, m, u)
    triples. Only n-1 comparisons are required. Comparative significances
    between consecutive ranked criteria (omega) and between criteria two
    ranks apart (phi, used to enforce transitivity) are derived via TFN
    division/multiplication, and the fully consistent fuzzy weights are
    obtained by solving a linear program that minimizes the maximum
    deviation from consistency (chi), subject to TFN ordering
    (l <= m <= u) and GMIR-based normalization.

    .. rubric:: Reference
        
        Pamucar, D., & Ecer, F. (2020). Prioritizing the weights
        of the evaluation criteria under fuzziness: the fuzzy full
        consistency method - FUCOM-F. Facta Universitatis, Series:
        Mechanical Engineering, 18(3), 419-437.

    Parameters
    ----------
    linguistic_scale : dict[str, tuple[float, float, float]] | None, optional
        Mapping from linguistic terms to TFNs ``(l, m, u)``, used to
        interpret string entries in `significance`. If None (default),
        `DEFAULT_SCALE` (the 5-point scale from Pamucar & Ecer, 2020) is
        used.
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.

    Attributes
    ----------
    chi : float | None
        The optimal consistency measure (deviation from maximum
        consistency) found by the optimizer, populated after `_calculate`
        runs.
    consistency : float | None
        Alias of `chi`, set via the inherited attribute for consistency
        with other subjective methods (`BaseSubjectiveFuzzyMethod`).

    Examples
    --------
    >>> from pyfdm.weights.subjective import fFUCOM
    >>> # C3 > C1 > C2 (original indices: C1=1, C2=2, C3=3)
    >>> order = [3, 1, 2]
    >>> significance = ['EI', 'WI', 'FI']
    >>> ffucom = fFUCOM()
    >>> weights = ffucom(criteria_order=order, significance=significance)
    >>> ffucom.chi
    """

    DEFAULT_SCALE = {
        'EI': (1, 1, 1),
        'WI': (2 / 3, 1, 3 / 2),
        'FI': (3 / 2, 2, 5 / 2),
        'VI': (5 / 2, 3, 7 / 2),
        'AI': (7 / 2, 4, 9 / 2),
    }

    def __init__(
        self,
        linguistic_scale: dict[str, tuple[float, float, float]] | None = None,
        logger: StepLogger | None = None
    ):
        super().__init__(logger=logger)

        self.linguistic_scale = (
            linguistic_scale
            if linguistic_scale is not None
            else self.DEFAULT_SCALE
        )

        self.chi: float | None = None

    def _to_tfn(
        self,
        value: str | tuple[float, float, float] | list[float] | np.ndarray
    ) -> tuple[float, float, float]:
        """
        Converts a linguistic term or a raw (l, m, u) triple into a TFN
        tuple.

        Parameters
        ----------
        value : str | tuple[float, float, float] | list[float] | np.ndarray
            Either a linguistic term present in `self.linguistic_scale`
            (case-insensitive), or a raw TFN as an (l, m, u) sequence.

        Returns
        -------
        tuple[float, float, float]
            The TFN as ``(l, m, u)``.

        Raises
        ------
        ValueError
            If `value` is a string not present in `self.linguistic_scale`,
            or a sequence that does not contain exactly three values, or
            cannot be converted to floats.
        """

        if isinstance(value, str):

            if value.upper() not in self.linguistic_scale:
                raise ValueError(
                    f"Unknown linguistic term {value}. "
                    f"Available: {list(self.linguistic_scale.keys())}"
                )

            return self.linguistic_scale[value.upper()]

        try:
            if len(value) != 3:
                raise ValueError("TFN must contain three values (l,m,u)")

            return tuple(map(float, value))
        except TypeError as e:
            raise ValueError(
                f"TFN must be a linguistic term or an (l, m, u) sequence, "
                f"got {type(value).__name__}."
            ) from e

    def _tfn_divide(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Approximate TFN division: ``a / b ≈ (a_l/b_u, a_m/b_m, a_u/b_l)``.

        Parameters
        ----------
        a : tuple[float, float, float]
            Dividend TFN ``(l, m, u)``.
        b : tuple[float, float, float]
            Divisor TFN ``(l, m, u)``.

        Returns
        -------
        tuple[float, float, float]
            The resulting TFN.
        """

        return (
            a[0] / b[2],
            a[1] / b[1],
            a[2] / b[0]
        )

    def _tfn_multiply(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        TFN multiplication: ``a (x) b = (a_l*b_l, a_m*b_m, a_u*b_u)``.

        Parameters
        ----------
        a : tuple[float, float, float]
            First TFN ``(l, m, u)``.
        b : tuple[float, float, float]
            Second TFN ``(l, m, u)``.

        Returns
        -------
        tuple[float, float, float]
            The resulting TFN.
        """

        return (
            a[0] * b[0],
            a[1] * b[1],
            a[2] * b[2]
        )

    def _calculate(
        self,
        criteria_order: list[int] | np.ndarray,
        significance: list[str | tuple[float, float, float]] | np.ndarray
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the FUCOM-F method.

        Parameters
        ----------
        criteria_order : list[int] | np.ndarray
            Ordered, 1-based indices of criteria, most important first
            (i.e. a permutation of ``1..n``).
        significance : list[str | tuple[float, float, float]] | np.ndarray
            Fuzzy significance for each ordered criterion (linguistic terms
            or raw (l, m, u) triples), given in the SAME order as
            `criteria_order`. The first entry is expected to be
            'EI' / (1, 1, 1).

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n, 3), as (l, m, u) triples, indexed by
            ORIGINAL criterion index (not by rank).

        Raises
        ------
        ValueError
            If `criteria_order` and `significance` have different lengths,
            if `criteria_order` is not a valid permutation of ``1..n``, or
            if any `significance` entry cannot be converted to a TFN.
        RuntimeError
            If the underlying linear program raises unexpectedly, or fails
            to converge (`result.success` is False).
        """

        def wl(i: int) -> int:
            return 3 * i

        def wm(i: int) -> int:
            return 3 * i + 1

        def wu(i: int) -> int:
            return 3 * i + 2

        try:
            order = np.asarray(criteria_order, dtype=int)
        except (TypeError, ValueError) as e:
            raise ValueError(f"'criteria_order' must be array-like integer data: {e}") from e

        n = len(order)

        if sorted(order.tolist()) != list(range(1, n + 1)):
            raise ValueError(
                "'criteria_order' must be a permutation of 1..n "
                f"(1-based criterion indices), got {order.tolist()}."
            )

        if len(significance) != n:
            raise ValueError("criteria_order and significance must have the same length")

        significance = [self._to_tfn(x) for x in significance]

        self._log_step(self.logger, "Criteria order", order)
        self._log_step(self.logger, "Significance TFNs", np.asarray(significance))

        omega = []
        omega_idx = []

        for pos in range(n - 1):

            i = order[pos] - 1
            j = order[pos + 1] - 1

            omega_idx.append((i, j))
            omega.append(self._tfn_divide(significance[pos + 1], significance[pos]))

        self._log_step(self.logger, "Comparative significances", np.asarray(omega))

        phi = []
        phi_idx = []

        for pos in range(n - 2):
            i = order[pos] - 1
            j = order[pos + 2] - 1

            phi_idx.append((i, j))
            phi.append(self._tfn_multiply(omega[pos], omega[pos + 1]))

        self._log_step(self.logger, "Full consistency significances", np.asarray(phi))

        nvar = 3 * n + 1
        chi_pos = nvar - 1

        A_ub = []
        b_ub = []

        def add_constraint(i: int, j: int, tfn: tuple[float, float, float]) -> None:
            """|w_i - w_j (x) tfn| <= chi, linearized per (l, m, u) component."""
            l, m, u = tfn

            pairs = [
                (wl(i), wu(j), l),
                (wm(i), wm(j), m),
                (wu(i), wl(j), u)
            ]

            for a, b, c in pairs:
                row = np.zeros(nvar)

                row[a] = 1
                row[b] = -c
                row[chi_pos] = -1

                A_ub.append(row)
                b_ub.append(0)

                row = np.zeros(nvar)

                row[a] = -1
                row[b] = c
                row[chi_pos] = -1

                A_ub.append(row)
                b_ub.append(0)

        for idx, tfn in zip(omega_idx, omega):
            add_constraint(*idx, tfn)

        for idx, tfn in zip(phi_idx, phi):
            add_constraint(*idx, tfn)

        # TFN ordering: w_i^l <= w_i^m <= w_i^u
        for i in range(n):
            row = np.zeros(nvar)
            row[wl(i)] = 1
            row[wm(i)] = -1

            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(nvar)
            row[wm(i)] = 1
            row[wu(i)] = -1

            A_ub.append(row)
            b_ub.append(0)

        # normalization: sum of GMIR(w_i) = 1
        A_eq = np.zeros((1, nvar))

        for i in range(n):
            A_eq[0, wl(i)] = 1 / 6
            A_eq[0, wm(i)] = 4 / 6
            A_eq[0, wu(i)] = 1 / 6

        b_eq = [1.0]
        bounds = [(1e-10, None)] * (3 * n) + [(0, None)]

        try:
            result = linprog(
                np.eye(1, nvar, chi_pos).flatten(),
                A_ub=np.asarray(A_ub),
                b_ub=np.asarray(b_ub),
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs"
            )
        except Exception as e:
            raise RuntimeError(f"fFUCOM linear program raised unexpectedly: {e}") from e

        if not result.success:
            raise RuntimeError(result.message)

        self.chi = result.x[-1]
        self.consistency = self.chi
        weights = result.x[:3 * n].reshape(n, 3)

        self._log_step(self.logger, "Fuzzy weights", weights)

        return weights