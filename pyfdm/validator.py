# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

__all__ = ['Validator']


class Validator:
    """
    Validation utilities for fuzzy methods.

    All methods raise descriptive ValueError or TypeError on failure,
    so callers receive actionable error messages.
    """

    @staticmethod
    def validate_matrix_shape(matrix: np.ndarray) -> None:
        """
        Verify that *matrix* is a 3-D array of shape (m, n, 3) where
        m >= 2 (at least two alternatives) and n >= 1.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f'Decision matrix must be a numpy ndarray, got {type(matrix).__name__}.'
            )
        if matrix.ndim != 3 or matrix.shape[2] != 3:
            raise ValueError(
                f'Decision matrix must have shape (m, n, 3) for Triangular Fuzzy Numbers, '
                f'got shape {matrix.shape}.'
            )
        if matrix.shape[0] < 2:
            raise ValueError(
                f'Decision matrix must contain at least 2 alternatives, '
                f'got {matrix.shape[0]}.'
            )
        if matrix.shape[1] < 1:
            raise ValueError('Decision matrix must contain at least 1 criterion.')

    @staticmethod
    def validate_tfn_values(matrix: np.ndarray) -> None:
        """
        Check that every TFN (l, m, u) in *matrix* satisfies l <= m <= u
        and contains no NaN or Inf values.
        """
        if np.any(np.isnan(matrix)):
            raise ValueError('Decision matrix contains NaN values.')
        if np.any(np.isinf(matrix)):
            raise ValueError('Decision matrix contains Inf values.')

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                l, m, u = matrix[i, j]
                if not (l <= m <= u):
                    raise ValueError(
                        f'Invalid TFN at alternative {i}, criterion {j}: '
                        f'requires l <= m <= u, got ({l}, {m}, {u}).'
                    )

    @staticmethod
    def validate_tfn(a: np.ndarray | list, name: str) -> None:
        """
        Validate and convert a Triangular Fuzzy Number.

        Parameters
        ----------
        a : np.ndarray | list 
            Triangular Fuzzy Number represented as (l, m, u).

        name : str
            Variable name used in error messages.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input cannot be converted to a numeric array or does not
            represent a valid Triangular Fuzzy Number.
        """

        try:
            a = np.asarray(a, dtype=float)

        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'{name}' must be a numeric Triangular Fuzzy Number: {e}"
            ) from e

        if a.shape != (3,):
            raise ValueError(
                f"'{name}' must have shape (3,), got {a.shape}."
            )

    @staticmethod
    def validate_weights(weights: np.ndarray, crisp_required: bool = False) -> None:
        """
        Validate criteria weights.
        Crisp (1-D): must sum to 1 and be non-negative.
        Fuzzy (2-D, shape n x 3): each row must be a valid TFN.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                f'Weights must be a numpy ndarray, got {type(weights).__name__}.'
            )
        if crisp_required:
            if weights.ndim != 1:
                raise ValueError(
                    'This method requires crisp (1-D) weights, '
                    f'got array of shape {weights.shape}.'
                )

        if weights.ndim == 1:
            if np.any(weights < 0):
                raise ValueError('Crisp weights must be non-negative.')
            total = np.sum(weights)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f'Crisp weights must sum to 1, got {total:.8f}.'
                )
        elif weights.ndim == 2:
            if weights.shape[1] != 3:
                raise ValueError(
                    f'Fuzzy weights must have shape (n, 3) for TFNs, '
                    f'got shape {weights.shape}.'
                )
            if np.any(weights < 0):
                raise ValueError('Fuzzy weight components must be non-negative.')
            for j in range(len(weights)):
                l, m, u = weights[j]
                if not (l <= m <= u):
                    raise ValueError(
                        f'Invalid TFN weight at criterion {j}: '
                        f'requires l <= m <= u, got ({l}, {m}, {u}).'
                    )
        else:
            raise ValueError(
                f'Weights must be 1-D (crisp) or 2-D (fuzzy TFN), '
                f'got {weights.ndim}-D array.'
            )

    @staticmethod
    def validate_types(
        types: np.ndarray | None,
        different_types: bool = False
    ) -> None:
        """
        Validate the criteria types vector.

        Parameters
        ----------
        types : np.ndarray | None
            Vector of criterion types. Each element should be:
            - ``1`` for a profit criterion,
            - ``-1`` for a cost criterion.
            If ``None``, validation is skipped.
        different_types : bool, default=False
            If ``True``, additionally requires the vector to contain at least
            one profit criterion and one cost criterion.

        Raises
        ------
        TypeError
            If ``types`` is not a NumPy array.
        ValueError
            If invalid criterion types are found or, when
            ``different_types=True``, all criteria are of the same type.
        """
        if types is None:
            return

        if not isinstance(types, np.ndarray):
            raise TypeError(
                f"Types must be a numpy ndarray, got {type(types).__name__}."
            )

        invalid = np.setdiff1d(np.unique(types), np.array([1, -1]))
        if invalid.size > 0:
            raise ValueError(
                "Criteria types must be 1 (profit) or -1 (cost), "
                f"found invalid values: {invalid.tolist()}."
            )

        if different_types:
            unique = np.unique(types)
            if not (1 in unique and -1 in unique):
                raise ValueError(
                    "The criteria types vector must contain at least one "
                    "profit criterion (1) and one cost criterion (-1)."
                )

    @staticmethod
    def validate_input(
        matrix: np.ndarray, 
        weights: np.ndarray | None = None,
        types: np.ndarray | None= None
    ) -> None:
        """Check dimensional consistency between matrix, weights, and types."""
        n_criteria = matrix.shape[1]
        if weights is not None:
            n_weights = weights.shape[0]
            if n_criteria != n_weights:
                raise ValueError(
                    f'Number of criteria ({n_criteria}) must equal '
                    f'number of weights ({n_weights}).'
                )
        if types is not None:
            n_types = types.shape[0]
            if n_criteria != n_types:
                raise ValueError(
                    f'Number of criteria ({n_criteria}) must equal '
                    f'number of types ({n_types}).'
                )

    @staticmethod
    def fuzzy_validation(
        matrix: np.ndarray, 
        weights: np.ndarray,
        types: np.ndarray,
        crisp_required: bool,
        different_types: bool
        ) -> None:
        """
        Run all standard validations for a fuzzy TFN MCDA problem.

        Checks (in order):
        1. matrix shape
        2. TFN value validity (l <= m <= u, no NaN/Inf)
        3. weights validity
        4. types validity
        5. dimensional consistency
        """
        Validator.validate_matrix_shape(matrix)
        Validator.validate_tfn_values(matrix)
        Validator.validate_weights(weights, crisp_required)
        Validator.validate_types(types, different_types)
        Validator.validate_input(matrix, weights, types)

    @staticmethod
    def validate_comparison_matrix(
        matrix: np.ndarray,
        dim: int = 3,
    ) -> None:
        """
        Validate a pairwise comparison matrix.

        Parameters
        ----------
        matrix : ndarray
            Pairwise comparison matrix.

        dim : {2, 3}, default=3
            Expected matrix dimensionality.
            - 2 : crisp comparison matrix of shape (n, n)
            - 3 : fuzzy comparison matrix of shape (n, n, 3)
        """

        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f"Comparison matrix must be a numpy ndarray, got {type(matrix).__name__}."
            )

        if dim not in (2, 3):
            raise ValueError("'dim' must be either 2 or 3.")

        if matrix.ndim != dim:
            raise ValueError(
                f"Comparison matrix must be {dim}-dimensional, got {matrix.ndim}."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Comparison matrix must be square (n x n), "
                f"got {matrix.shape[0]} × {matrix.shape[1]}."
            )

        if dim == 3 and matrix.shape[2] != 3:
            raise ValueError(
                f"Fuzzy comparison matrix must have shape (n, n, 3), got {matrix.shape}."
            )

    @staticmethod
    def validate_param_range(
        value: int | float,
        minimum: int | float,
        maximum: int | float,
        parameter_name: str
    ) -> None:
        """
        Validate whether a numeric parameter belongs to a specified interval.

        Parameters
        ----------
        value : int | float
            Value to validate.
        minimum : int | float
            Lower bound of the interval.
        maximum : int | float
            Upper bound of the interval.
        parameter_name : str
            Name of the validated parameter.

        Raises
        ------
        TypeError
            If the parameter is not numeric.
        ValueError
            If the value lies outside the specified interval.
        """
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"'{parameter_name}' must be a numeric value, "
                f"got {type(value).__name__}."
            )

        if not (minimum <= value <= maximum):
            raise ValueError(f"'{parameter_name}' must be in [{minimum}, {maximum}], got {value}.")

    @staticmethod
    def validate_vectors(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and coerce two input vectors.

        Parameters
        ----------
        x : np.ndarray
            First input vector.

        y : np.ndarray
            Second input vector.

        ranking : bool, default=False
            If True, validates that the vectors represent rankings
            (positive integers without duplicates).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Validated float numpy arrays.

        Raises
        ------
        ValueError
            If the vectors cannot be converted to float arrays, have different
            lengths, are not one-dimensional, are empty, or (when
            ``ranking=True``) are not valid rankings.
        """

        try:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'x' and 'y' must be numeric vectors convertible to "
                f"numpy arrays: {e}"
            ) from e

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                f"'x' and 'y' must be one-dimensional arrays, "
                f"got {x.ndim}D and {y.ndim}D."
            )

        if x.shape != y.shape:
            raise ValueError(
                f"'x' and 'y' must have the same length, "
                f"got {len(x)} and {len(y)}."
            )

        if x.size == 0:
            raise ValueError("Input vectors cannot be empty.")

        return x, y

    @staticmethod
    def validate_ervd_input(
        ref_point: np.ndarray | list | None,
        n: int
    ) -> np.ndarray | None:
        """
        Validate the ERVD reference point.

        Parameters
        ----------
        ref_point : np.ndarray | list | None
            Reference point represented as Triangular Fuzzy Numbers.
        n : int
            Number of criteria in decision matrix

        Returns
        -------
        np.ndarray | None
            Validated reference point converted to a NumPy array.

        Raises
        ------
        ValueError
            If the reference point cannot be converted to a numeric array
            or does not have shape ``(n, 3)``.
        """
        if ref_point is None:
            return None

        try:
            ref_point = np.asarray(ref_point, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "'ref_point' must be array-like numeric data "
                f"convertible to a float numpy array: {e}"
            ) from e

        if ref_point.ndim != 2 or ref_point.shape[1] != 3:
            raise ValueError(
                f"'ref_point' must have shape (n, 3), got {ref_point.shape}."
            )

        if ref_point is not None and ref_point.shape[0] != n:
            raise ValueError(
                f"'ref_point' has {ref_point.shape[0]} criteria, but "
                f"'matrix' has {n}."
            )

        return ref_point

    @staticmethod
    def validate_rim_input(
        matrix: np.ndarray,
        lower_bound: np.ndarray | None = None,
        upper_bound: np.ndarray | None = None,
        lower_reference: np.ndarray | None = None,
        upper_reference: np.ndarray | None = None,
    ) -> None:
        def _validate_bounds(bounds: np.ndarray | None, name: str) -> np.ndarray | None:
            """Validates an optional (n, 3) array of per-criterion TFN bounds."""
            if bounds is None:
                return None

            try:
                bounds = np.asarray(bounds, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"'{name}' must be array-like numeric data convertible to "
                    f"a float numpy array: {e}"
                ) from e

            if matrix.shape[1] != bounds.shape[0]:
                raise ValueError(
                    f"'{name}' is inconsistent with the decision matrix. "
                    f"Expected {matrix.shape[1]} criteria, got {bounds.shape[0]}."
                )
                
            if bounds.ndim != 2 or bounds.shape[-1] != 3:
                raise ValueError(f"'{name}' must have shape (n, 3), got {bounds.shape}.")

        _validate_bounds(lower_bound, 'lower_bound')
        _validate_bounds(upper_bound, 'upper_bound')
        _validate_bounds(lower_reference, 'lower_reference')
        _validate_bounds(upper_reference, 'upper_reference')

    @staticmethod
    def validate_rafsi_input(
        matrix: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        ideal: np.ndarray | None,
        anti_ideal: np.ndarray | None,
    ) -> None:
        def _validate_interval(lower_bound: float, upper_bound: float) -> None:
                """Validates the mapping-interval bounds lower_bound < upper_bound."""
                if lower_bound >= upper_bound:
                    raise ValueError(f"'lower_bound' must be strictly less than 'upper_bound', got lower_bound={lower_bound}, upper_bound={upper_bound}.")

        def _validate_bounds(
            bounds: np.ndarray | None,
            name: str,
            n_criteria: int
        ) -> np.ndarray | None:

            if bounds is None:
                return None

            try:
                bounds = np.asarray(bounds, dtype=float)

            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"'{name}' must be array-like numeric data convertible to "
                    f"a float numpy array: {e}"
                ) from e

            if bounds.ndim != 1:
                raise ValueError(f"'{name}' must be a 1D array, got shape {bounds.shape}.")

            if bounds.shape[0] != n_criteria:
                raise ValueError(
                    f"'{name}' length must match the number of criteria. "
                    f"Expected {n_criteria}, got {bounds.shape[0]}."
                )

        _validate_interval(lower_bound, upper_bound)
        _validate_bounds(ideal, 'ideal', matrix.shape[1])
        _validate_bounds(anti_ideal, 'anti_ideal', matrix.shape[1])

    @staticmethod
    def validate_spotis_input(
        matrix: np.ndarray,
        bounds: np.ndarray | None
    ) -> None:
        """
        Validate SPOTIS criterion bounds.

        Parameters
        ----------
        matrix : np.ndarray
            Decision matrix.
        bounds : np.ndarray | None
            Criterion bounds of shape (n, 2), where n is the number of
            criteria.

        Raises
        ------
        ValueError
            If `bounds` has an invalid shape or an inconsistent number of
            criteria.
        """

        if bounds is None:
            return

        try:
            bounds = np.asarray(bounds, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'bounds' must be array-like numeric data convertible "
                f"to a float numpy array: {e}"
            ) from e

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"'bounds' must have shape (n, 2), got {bounds.shape}.")

        if bounds.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"'bounds' must contain one row per criterion. "
                f"Expected {matrix.shape[1]}, got {bounds.shape[0]}."
            )

    @staticmethod
    def validate_lmaw_input(
        expert_decisions,
        linguistic_scale: dict | None = None,
    ) -> None:
        """
        Validate experts' decisions.

        Accepted formats
        ----------------
        Linguistic:
            (n_criteria,)
            (n_experts, n_criteria)

        TFNs:
            (n_criteria, 3)
            (n_experts, n_criteria, 3)
        """

        arr = np.asarray(expert_decisions)
        first = arr.flat[0]

        if not isinstance(first, str):
            # ---------- TFNs ----------
            if arr.ndim == 2 and arr.shape[-1] == 3:
                if not np.issubdtype(np.asarray(arr).dtype, np.number):
                    raise TypeError("Triangular fuzzy numbers must contain only numeric values.")
                return

            if arr.ndim == 3 and arr.shape[-1] == 3:
                if not np.issubdtype(np.asarray(arr).dtype, np.number):
                    raise TypeError("Triangular fuzzy numbers must contain only numeric values.")
                return

        # ---------- Linguistic ----------
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim == 2:
            if linguistic_scale is None:
                raise ValueError("A linguistic_scale must be provided when using linguistic terms.")

            allowed = set(linguistic_scale.keys())

            for expert_idx, expert in enumerate(arr):
                for crit_idx, token in enumerate(expert):

                    if token not in allowed:
                        raise ValueError(
                            f"Unknown linguistic term '{token}' "
                            f"for expert {expert_idx + 1}, "
                            f"criterion {crit_idx + 1}. "
                            f"Allowed values are: {sorted(allowed)}."
                        )

            return

        raise ValueError(
            "Invalid expert_decisions shape. Expected one of:\n"
            "  (n_criteria,)\n"
            "  (n_experts, n_criteria)\n"
            "  (n_criteria, 3)\n"
            "  (n_experts, n_criteria, 3)"
        )

    @staticmethod
    def validate_bwm_input(
        best_to_others,
        others_to_worst,
        best_idx,
        worst_idx,
    ):

        if best_to_others.ndim != 2 or best_to_others.shape[1] != 3:
            raise ValueError("best_to_others must have shape (n,3)")

        if others_to_worst.shape != best_to_others.shape:
            raise ValueError("others_to_worst must have shape (n,3)")

        n = best_to_others.shape[0]

        if not (0 <= best_idx < n):
            raise ValueError("Incorrect best_idx.")

        if not (0 <= worst_idx < n):
            raise ValueError("Incorrect worst_idx.")

