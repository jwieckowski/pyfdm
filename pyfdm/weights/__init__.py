# Copyright (c) 2026 Jakub Więckowski

from .objective import (
    equal_weights,
    shannon_entropy_weights,
    standard_deviation_weights,
    variance_weights,
)
from . import objective
from . import subjective

__all__ = [
    # objective (direct re-export for API compatibility)
    'equal_weights',
    'shannon_entropy_weights',
    'standard_deviation_weights',
    'variance_weights',
    # submodules
    'objective',
    'subjective',
]
