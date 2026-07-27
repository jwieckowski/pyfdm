# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from ..step_logger import StepLogger
from ..validator import Validator
from ..utils.ranking import rank_alternatives

__all__ = ['BaseFuzzyMethod']

class BaseFuzzyMethod(ABC):
    """
    Abstract base class for fuzzy MCDA methods based on
    Triangular Fuzzy Numbers (TFNs).

    This class provides common functionality for fuzzy decision-making
    methods, including:

    - execution wrapper via ``__call__``,
    - result validation,
    - intermediate step storage,
    - optional logging,
    - alternative ranking.

    Subclasses must implement the ``_calculate`` method responsible for
    the specific MCDA computation.

    Attributes
    ----------
    preferences : np.ndarray | None
        Final preference values of alternatives.

    intermediate_results : dict[str, dict]
        Stored intermediate calculation steps.

    logger : StepLogger | None
        Optional logger used for recording computation steps.

    _descending : bool
        Determines ranking direction. If True, larger preference values
        indicate better alternatives.
    """

    _descending: bool = True
    _crisp_weights_required: bool = False
    _different_types_required: bool = False

    def __init__(self, logger: StepLogger | None = None):
        self.logger = logger
        self.preferences: np.ndarray = None
        self.intermediate_results: dict = {}
        
    def __call__(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute fuzzy MCDA method.

        Parameters
        ----------
        matrix : np.ndarray | list
            Fuzzy decision matrix with shape (m,n,3).

        weights : np.ndarray | list
            Criterion weights with shape (n,) or (n,3).

        types : np.ndarray | list
            Criterion types:
            1 - profit criterion,
            -1 - cost criterion.

        Returns
        -------
        np.ndarray
            Preference values of alternatives.

        Raises
        ------
        ValueError
            If input data is invalid.

        RuntimeError
            If calculation or logging fails.
        """

        self.preferences = None
        self.intermediate_results.clear()

        self._validate_input(
            matrix,
            weights,
            types,
            *args,
            **kwargs
        )

        self.logger = self._prepare_logger(self.logger)

        try:
            preferences = self._calculate(
                matrix,
                weights,
                types,
                *args,
                **kwargs
            )

        except Exception as e:
            raise RuntimeError(f"'{self.method_name}' failed during calculation: {e}") from e

        try:
            preferences = np.asarray(preferences, dtype=float)

        except (TypeError, ValueError) as e:
            raise TypeError(
                f"'{self.method_name}._calculate' must return "
                f"numeric array-like data."
            ) from e


        if self.logger is not None:
            try:
                if "console" in self.logger.output:
                    self.logger._write_console()

                self.logger.save()

            except Exception as e:
                raise RuntimeError(f"Logger failed for '{self.method_name}': {e}") from e

        self.preferences = preferences

        return preferences

    @abstractmethod
    def _calculate(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray | list:
        """
        Perform method-specific fuzzy MCDA calculation.

        Must be implemented by subclasses.

        Returns
        -------
        np.ndarray | list
            Preference values.
        """
        pass

    def _validate_input(
        self,
        matrix: np.ndarray | list,
        weights: np.ndarray | list,
        types: np.ndarray | list,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Perform common validation of fuzzy MCDA inputs.

        This method is intentionally designed to be overridden by
        subclasses requiring additional validation.

        Parameters
        ----------
        matrix : np.ndarray | list
            Decision matrix.

        weights : np.ndarray | list
            Criterion weights.

        types : np.ndarray | list | None
            Criteria types, optional, default = None

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
            Validator.fuzzy_validation(
                matrix, 
                weights,
                types, 
                self._crisp_weights_required,
                self._different_types_required
            )

        except Exception as e:
            raise e

    def rank(self) -> np.ndarray:
        """
        Calculate the alternatives ranking based on self.preferences.

        Must be called after __call__().

        Returns
        -------
            ndarray, shape (m,)
                Ranking positions (1 = best).

        Raises
        ------
            AttributeError
                If called before the method has been evaluated.
        """
        if self.preferences is None:
            raise AttributeError(f'{self.method_name}: call the method first before requesting a ranking.')

        return rank_alternatives(self.preferences, descending=self._descending, method='average')

    @property
    def method_name(self) -> str:
        """Return the class name of this method."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Returns a compact, debugger/log-friendly representation of the
        method instance.

        Every public attribute (i.e. not starting with '_') other than
        `preferences` is included. Numpy arrays are shown as their shape
        rather than their full contents, and callables are shown by name,
        to keep the representation short and readable.

        Returns
        -------
        str
            String of the form ``MethodName(param1=value1, param2=value2, ...)``.
        """
        def _repr_value(v: Any) -> str:
            try:
                if isinstance(v, np.ndarray):
                    return f'array(shape={v.shape})'
                if callable(v):
                    return v.__name__
                return repr(v)
            except Exception:
                return f'<unrepresentable {type(v).__name__}>'

        params = ', '.join(
            f'{k}={_repr_value(v)}'
            for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'preferences'
        )
        return f'{self.method_name}({params})'

    def _prepare_logger(self, logger: StepLogger | None) -> StepLogger | None:
        """
        Validates the given logger and binds it to this method instance.

        Parameters
        ----------
        logger : StepLogger | None
            Logger to prepare. If None, logging is skipped entirely.

        Returns
        -------
        StepLogger | None
            The same logger, with its method name set to `self.method_name`,
            or None if no logger was given.

        Raises
        ------
        TypeError
            If `logger` is not None and not a `StepLogger` instance.
        RuntimeError
            If binding the method name to the logger fails.
        """

        if logger is None:
            return None

        if not isinstance(logger, StepLogger):
            raise TypeError(
                f"'logger' must be a StepLogger instance or None, got "
                f"{type(logger).__name__}."
            )

        try:
            logger._set_method(self.method_name)
        except Exception as e:
            raise RuntimeError(f"Failed to bind logger to method '{self.method_name}': {e}") from e

        return logger

    def _store_step(
        self,
        name: str,
        data: np.ndarray | int | float | str | dict | list,
        description: str = ''
    ) -> None:
        """
        Store intermediate computation result.

        Parameters
        ----------
        name : str
            Step name.
        data : np.ndarray | int | float | str | dict | list
            Intermediate result.
        description : str, optional
            Optional explanation. If empty, the 'description' key is
            omitted from the stored entry.

        Raises
        ------
        ValueError
            If `data` cannot be serialized into a step record (e.g. a numpy
            array containing non-serializable objects).
        """

        try:
            if isinstance(data, np.ndarray):
                entry = {
                    'shape': list(data.shape),
                    'data': data.tolist()
                }
            else:
                entry = {
                    'data': data
                }
        except Exception as e:
            raise ValueError(
                f"Failed to store intermediate step '{name}' for "
                f"'{self.method_name}': {e}"
            ) from e

        if description:
            entry['description'] = description

        self.intermediate_results[name] = entry

    def _log_step(
        self,
        logger: StepLogger | None,
        name: str,
        data: np.ndarray | int | float | str | dict | list,
        description: str = ''
    ) -> None:
        """
        Records an intermediate computation step, both into
        `intermediate_results` (always) and into `logger` (if provided).

        Parameters
        ----------
        logger : StepLogger | None
            Logger to forward the step to. Typically the value returned by
            `_prepare_logger`. If None, the step is only stored in
            `intermediate_results`.
        name : str
            Step name (e.g. 'Comparison matrix', 'Geometric mean').
        data : np.ndarray | int | float | str | dict | list
            Intermediate result to record.
        description : str, optional
            Optional human-readable explanation of the step.

        Raises
        ------
        ValueError
            If `data` cannot be stored (propagated from `_store_step`).
        RuntimeError
            If the logger fails while recording the step.
        """
        self._store_step(name, data, description)

        if logger is not None:
            try:
                logger.log(name, data, description)
            except Exception as e:
                raise RuntimeError(
                    f"Logger failed while recording step '{name}' for "
                    f"'{self.method_name}': {e}"
                ) from e