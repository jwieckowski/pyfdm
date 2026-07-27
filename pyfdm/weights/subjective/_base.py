# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from ...step_logger import StepLogger

class BaseSubjectiveFuzzyMethod(ABC):
    """
    Abstract base class for subjective fuzzy criteria-weighting methods

    Subclasses implement the `_calculate` method, which performs the actual
    weight-determination logic and returns a fuzzy weight vector. This base
    class takes care of the shared, method-agnostic concerns: coercing the
    result to a numpy array, optional step-by-step logging via a
    `StepLogger`, and a uniform string representation.

    Parameters
    ----------
    logger : StepLogger | None, optional
        Logger instance used to record intermediate computation steps
        (see `_log_step`). If None (default), no logging is performed and
        `_log_step`/`_store_step` calls only populate `intermediate_results`.

    Attributes
    ----------
    consistency : float | None
        Consistency/deviation measure of the derived weights (e.g. a
        consistency ratio or a deviation-from-maximum-consistency value),
        populated by subclasses that support such a check. None if the
        method does not compute one or `_calculate` has not been called yet.
    intermediate_results : dict[str, dict]
        Mapping of step name -> step record (as stored by `_store_step`),
        populated during `_calculate` regardless of whether a logger is
        attached.

    """

    def __init__(self, logger: StepLogger | None = None):
        self.logger = logger
        self.consistency: float | None = None
        self.intermediate_results: dict[str, dict] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Runs the weighting method and returns the fuzzy weight vector.

        Prepares the attached logger (if any), delegates the actual
        computation to `_calculate`, coerces its result to a numpy array of
        floats, and flushes the logger (console output and/or file save).

        Parameters
        ----------
        *args : Any
            Forwarded as-is to `_calculate`; see the concrete subclass for
            the expected signature (e.g. a comparison matrix, a criteria
            ranking, or linguistic significance values).
        **kwargs : Any
            Forwarded as-is to `_calculate`.

        Returns
        -------
        np.ndarray
            The fuzzy weight vector/matrix returned by `_calculate`, cast to
            `float` dtype.

        Raises
        ------
        TypeError
            If `self.logger` is set but is not a `StepLogger` instance, or
            if `_calculate` returns data that cannot be converted to a
            float numpy array.
        RuntimeError
            If `_calculate` raises during computation, or if the logger
            fails while writing console output or saving.
        """

        self.consistency = None
        self.intermediate_results.clear()

        logger = self._prepare_logger(self.logger)

        try:
            fuzzy_weights = self._calculate(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"'{self.method_name}' failed while computing fuzzy weights: {e}") from e

        try:
            fuzzy_weights = np.asarray(fuzzy_weights, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"'{self.method_name}._calculate' must return array-like "
                f"numeric data convertible to a float numpy array, got "
                f"{type(fuzzy_weights).__name__}."
            ) from e

        if logger is not None:
            try:
                if 'console' in logger.output:
                    logger._write_console()
                logger.save()
            except Exception as e:
                raise RuntimeError(
                    f"Logger failed while writing/saving output for "
                    f"'{self.method_name}': {e}"
                ) from e

        return fuzzy_weights

    @abstractmethod
    def _calculate(self, *args: Any, **kwargs: Any) -> np.ndarray | list:
        """
        Computes the fuzzy weights for this method. Must be implemented by
        every subclass.

        This is the only method a subclass is required to implement; all
        method-specific logic (building comparison matrices, solving
        optimization models, etc.) belongs here. Use
        `_log_step`/`_store_step` to record intermediate results as the
        computation progresses.

        Parameters
        ----------
        *args : Any
            Method-specific inputs (e.g. a comparison matrix and criteria
            labels, or a criteria ranking and significance values).
        **kwargs : Any
            Method-specific keyword inputs.

        Returns
        -------
        array_like
            Fuzzy weights, typically of shape (n_criteria, 3) as (l, m, u)
            triples. Coerced to a float numpy array by `__call__`.
        """
        pass

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