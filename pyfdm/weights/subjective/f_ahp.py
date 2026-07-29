# Copyright (c) 2026 Jakub Więckowski

import numpy as np
import warnings

from ._base import BaseSubjectiveFuzzyMethod
from ...step_logger import StepLogger
from ...validator import Validator

__all__ = ['fAHP']

class fAHP(BaseSubjectiveFuzzyMethod):
    """
    Fuzzy Analytic Hierarchy Process (F-AHP).

    Fuzzy AHP extends Saaty's AHP to a fuzzy environment: pairwise comparisons
    of criteria are expressed using linguistic terms mapped to Triangular
    Fuzzy Numbers (TFN) instead of crisp values. Fuzzy weights are derived
    via the extent-analysis-free geometric mean method (fuzzy geometric
    mean of each row of the comparison matrix, normalized by the reversed
    sum of all row geometric means), and consistency can optionally be
    checked on the defuzzified (crisp) comparison matrix and weights,
    analogous to Saaty's consistency ratio.

    Attributes
    ----------
    scale : dict[str, list[float]]
        The linguistic scale in effect (either the one passed in, or
        `DEFAULT_SCALE`).
    consistency : float | None
        Defuzzified consistency ratio computed by `_check_consistency`, if
        `consistency_check` is True; otherwise None (inherited from
        `BaseSubjectiveFuzzyMethod`).

    Parameters
    ----------
    scale : dict[str, list[float]] | None, optional
        Mapping from linguistic terms to TFNs ``[l, m, u]``, used both to
        interactively build a comparison matrix (see `_create_matrix`) and
        to validate/interpret an externally supplied one. If None
        (default), `DEFAULT_SCALE` (a 9-point Saaty-style linguistic scale)
        is used.
    consistency_check : bool, default=False
        Whether to compute a defuzzified (Saaty-style) consistency ratio
        after deriving the weights, and warn if it exceeds 0.10.
    logger : StepLogger | None, optional
        Step logger used to record intermediate computation steps.


    References
    ----------
    Sun, C. C. (2010). A performance evaluation model by integrating 
    fuzzy AHP and fuzzy TOPSIS methods. Expert systems with applications,
    37(12), 7745-7754.

    Examples
    --------
    >>> import numpy as np
    >>> from pyfdm.weights.subjective import fAHP
    >>> # 3x3x3 fuzzy comparison matrix (l, m, u) for 3 criteria
    >>> comp = np.array([
    ...     [[1, 1, 1], [1, 2, 3], [2, 3, 4]],
    ...     [[1/3, 1/2, 1], [1, 1, 1], [1, 2, 3]],
    ...     [[1/4, 1/3, 1/2], [1/3, 1/2, 1], [1, 1, 1]],
    ... ])
    >>> fahp = fAHP(consistency_check=True)
    >>> weights = fahp(comparison_matrix=comp)
    """

    DEFAULT_SCALE = {
        'Equal': [1, 1, 1],
        'Slightly more important': [1, 2, 3],
        'Weakly important': [2, 3, 4],
        'Between weekly and fairly important': [3, 4, 5],
        'Fairly important': [4, 5, 6],
        'Between fairly and strongly important': [5, 6, 7],
        'Strongly important': [6, 7, 8],
        'Between strongly and absolutely important': [7, 8, 9],
        'Absolutely important': [9, 9, 9]
    }

    def __init__(
        self,
        scale: dict[str, list[float]] | None = None,
        consistency_check: bool = False,
        logger: StepLogger | None = None,
    ):
        super().__init__(logger=logger)
        self.scale = scale if scale is not None else self.DEFAULT_SCALE
        self.consistency_check = consistency_check

    def _calculate(
        self,
        comparison_matrix: list[list[list[int | float]]] | np.ndarray | None = None,
        crit_labels: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Computes fuzzy criteria weights using the F-AHP method.

        Either `comparison_matrix` must be supplied directly, or
        `crit_labels` must be supplied so that a comparison matrix can be
        built interactively (see `_create_matrix`).

        Parameters
        ----------
        comparison_matrix : list[list[list[int | float]]] | np.ndarray | None, optional
            Fuzzy pairwise comparison matrix of shape (n, n, 3), as (l, m, u)
            triples. If None, `crit_labels` is used to build one
            interactively.
        crit_labels : list[str] | np.ndarray | None, optional
            Criterion labels, used only when `comparison_matrix` is None.

        Returns
        -------
        np.ndarray
            Fuzzy weights of shape (n, 3), as (l, m, u) triples.

        Raises
        ------
        ValueError
            If neither `comparison_matrix` nor `crit_labels` is provided,
            or if `comparison_matrix` fails validation.
        RuntimeError
            If the fuzzy geometric mean / weight computation fails (e.g.
            due to a malformed comparison matrix).
        """

        if comparison_matrix is None and crit_labels is None:
            raise ValueError("Either 'comparison_matrix' or 'crit_labels' must be provided.")

        comp = self._create_matrix(crit_labels) if comparison_matrix is None else comparison_matrix

        try:
            comp = np.asarray(comp, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'comparison_matrix' must be array-like numeric data "
                f"convertible to a float numpy array: {e}"
            ) from e

        try:
            Validator.validate_comparison_matrix(comp)
        except Exception as e:
            raise ValueError(f"Invalid comparison matrix: {e}") from e

        self._log_step(self.logger, "Comparison matrix", comp)

        try:
            # Fuzzy geometric mean
            geo_mean = []
            for i in range(3):
                gmean = np.prod(comp[:, :, i], axis=1) ** (1 / comp.shape[0])
                geo_mean.append(gmean)
            geo_mean = np.array(geo_mean).T

            self._log_step(self.logger, "Geometric mean", geo_mean)

            # sum of geometric mean weights
            geom_mean_sum = np.sum(geo_mean, axis=0)

            self._log_step(self.logger, "Sum of geometric mean", geom_mean_sum)

            # reciprocal of the sum of geometric mean weights
            rev = (geom_mean_sum ** -1)[::-1]

            self._log_step(self.logger, "Reciprocal of geometric mean sum", rev)

            # relative fuzzy weights
            rfw = geo_mean * rev

            self._log_step(self.logger, "Relative fuzzy weights", rfw)
        except Exception as e:
            raise RuntimeError(f"Failed to compute F-AHP weights: {e}") from e

        if self.consistency_check:
            try:
                self._check_consistency(comp, rfw)
            except Exception as e:
                raise RuntimeError(f"Consistency check failed: {e}") from e

        return rfw

    def _create_matrix(self, crit_labels: list[str] | np.ndarray) -> np.ndarray:
        """
        Create a fuzzy pairwise comparison matrix interactively.

        For each pair of criteria, prompts the user to pick the more
        important one and then the linguistic degree of preference from
        `self.scale`, storing the corresponding TFN (and its reciprocal in
        the mirrored position).

        Parameters
        ----------
        crit_labels : list[str] | np.ndarray
            Criterion labels.

        Returns
        -------
        np.ndarray
            Fuzzy pairwise comparison matrix of shape (n, n, 3).

        Raises
        ------
        ValueError
            If `crit_labels` is empty or cannot be converted to an array.
        """

        try:
            crit_labels = np.asarray(crit_labels)
        except (TypeError, ValueError) as e:
            raise ValueError(f"'crit_labels' must be array-like: {e}") from e

        n = len(crit_labels)
        if n == 0:
            raise ValueError("'crit_labels' must contain at least one criterion.")

        matrix = np.zeros((n, n, 3), dtype=float)

        # diagonal
        for i in range(n):
            matrix[i, i] = (1.0, 1.0, 1.0)

        scale_items = list(self.scale.items())

        print("\nAvailable linguistic scale:")
        for idx, (label, tfn) in enumerate(scale_items, start=1):
            print(f"{idx:>2}. {label:<40} {list(tfn)}")
        print()

        for i in range(n):
            for j in range(i):

                # determine preferred criterion
                while True:
                    answer = input(
                        f"Is '{crit_labels[j]}' more important than '{crit_labels[i]}'? [y/n]: "
                    ).strip().lower()

                    if answer in ("y", "n"):
                        break

                    print("Please enter 'y' or 'n'.\n")

                if answer == "y":
                    preferred = crit_labels[j]
                    less = crit_labels[i]
                else:
                    preferred = crit_labels[i]
                    less = crit_labels[j]

                # choose linguistic value
                while True:

                    print(f"\nHow much more important is '{preferred}' than '{less}'?")

                    for idx, (label, tfn) in enumerate(scale_items, start=1):
                        print(f"{idx:>2}. {label:<40} {tuple(tfn)}")

                    try:
                        choice = int(input("Select scale value: "))

                        if 1 <= choice <= len(scale_items):
                            break

                    except ValueError:
                        pass

                    print(f"Please enter an integer between 1 and {len(scale_items)}.\n")

                _, tfn = scale_items[choice - 1]

                tfn = np.asarray(tfn, dtype=float)
                reciprocal = 1.0 / tfn[::-1]

                if answer == "y":
                    matrix[j, i] = tfn
                    matrix[i, j] = reciprocal
                else:
                    matrix[j, i] = reciprocal
                    matrix[i, j] = tfn

        return matrix

    def _check_consistency(self, comp: np.ndarray, weights: np.ndarray) -> None:
        """
        Compute a defuzzified (Saaty-style) consistency ratio and warn if
        it exceeds the acceptable threshold.

        The fuzzy comparison matrix and fuzzy weights are first defuzzified
        via the Graded Mean Integration Representation (GMIR),
        ``R(a) = (l + 4m + u) / 6``, and the standard AHP eigenvalue-based
        consistency ratio (CI/RI) is then computed on the crisp values.

        Parameters
        ----------
        comp : np.ndarray
            Fuzzy comparison matrix of shape (n, n, 3).
        weights : np.ndarray
            Fuzzy weights of shape (n, 3).

        Raises
        ------
        ValueError
            If `comp`/`weights` have incompatible or unexpected shapes.

        Notes
        -----
        Sets `self.consistency` to the computed ratio. If it exceeds 0.10,
        a `UserWarning` is issued (the result is still returned, not
        blocked, since 0.10 is a conventional rather than a hard cutoff).
        """

        if comp.ndim != 3 or comp.shape[0] != comp.shape[1] or comp.shape[2] != 3:
            raise ValueError(f"'comp' must have shape (n, n, 3), got {comp.shape}.")

        n = comp.shape[0]

        crisp_comp = (
            comp[:, :, 0]
            + 4 * comp[:, :, 1]
            + comp[:, :, 2]
        ) / 6

        crisp_weights = (
            weights[:, 0]
            + 4 * weights[:, 1]
            + weights[:, 2]
        ) / 6

        crisp_weights /= np.sum(crisp_weights)

        Aw = crisp_comp @ crisp_weights

        lambda_max = np.mean(
            Aw / np.maximum(crisp_weights, 1e-10)
        )

        CI = (lambda_max - n) / (n - 1) if n > 1 else 0

        RI_TABLE = {
            1: 0.00,
            2: 0.00,
            3: 0.58,
            4: 0.90,
            5: 1.12,
            6: 1.24,
            7: 1.32,
            8: 1.41,
            9: 1.45,
            10: 1.49,
        }

        RI = RI_TABLE.get(n, 1.49)

        self.consistency = CI / RI if RI > 0 else 0

        if self.consistency > 0.10:
            warnings.warn(
                f"Consistency Ratio ({self.consistency:.4f}) exceeds the "
                f"acceptable threshold (0.10)."
            )