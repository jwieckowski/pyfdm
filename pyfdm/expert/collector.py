# Copyright (c) 2026 Jakub Więckowski


import numpy as np
from .scales import SCALE_1_9, get_tfn

__all__ = ['ExpertCollector']


class ExpertCollector:
    """
    Collect expert decision matrices and convert them to TFN form.

    Experts provide integer ratings using a linguistic scale.
    The collector converts each rating to a Triangular Fuzzy Number
    using the chosen scale.

    Parameters
    ----------
        n_alternatives : int
            Number of alternatives (rows in decision matrix).

        n_criteria : int
            Number of criteria (columns in decision matrix).

        scale : dict, default SCALE_1_9
            Mapping from integer rating to (l, m, u). Use one of
            the built-in scales (SCALE_1_5, SCALE_1_7, SCALE_1_9)
            or provide a custom dict.

    Examples
    --------
        >>> from pyfdm.expert import ExpertCollector
        >>> from pyfdm.expert.scales import SCALE_1_9
        >>> collector = ExpertCollector(n_alternatives=4, n_criteria=3, scale=SCALE_1_9)
        >>> collector.add_expert_matrix(0, [[9, 3, 7], [5, 9, 3], [3, 5, 9], [7, 3, 5]])
        >>> matrices = collector.get_tfn_matrices()
    """

    def __init__(self, n_alternatives: int, n_criteria: int, scale: dict = SCALE_1_9):
        if n_alternatives < 2:
            raise ValueError('n_alternatives must be at least 2.')
        if n_criteria < 1:
            raise ValueError('n_criteria must be at least 1.')

        self.n_alternatives = n_alternatives
        self.n_criteria = n_criteria
        self.scale = scale
        self._raw: dict[int, np.ndarray] = {}      # expert_id → int matrix (m, n)
        self._tfn: dict[int, np.ndarray] = {}      # expert_id → TFN matrix (m, n, 3)

    #  Adding expert data                                                  
    def add_expert_matrix(
        self, 
        expert_id: int,
        ratings: list | np.ndarray
    ) -> None:
        """
        Add a decision matrix from one expert.

        Parameters
        ----------
            expert_id : int
                Unique identifier for this expert (0-based or any int).

            ratings : list or ndarray, shape (m, n)
                Integer ratings in the chosen scale.

        Raises
        ------
            ValueError - on shape mismatch or invalid rating values.
        """
        ratings = np.asarray(ratings, dtype=int)

        if ratings.shape != (self.n_alternatives, self.n_criteria):
            raise ValueError(
                f'Expected ratings of shape ({self.n_alternatives}, {self.n_criteria}), '
                f'got {ratings.shape}.'
            )

        # Validate all values are in scale
        valid = set(self.scale.keys())
        invalid = set(ratings.flatten().tolist()) - valid
        if invalid:
            raise ValueError(
                f'Expert {expert_id}: ratings contain values not in scale: '
                f'{sorted(invalid)}. Valid: {sorted(valid)}.'
            )

        self._raw[expert_id] = ratings

        # Convert to TFN
        tfn_matrix = np.zeros((self.n_alternatives, self.n_criteria, 3))
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                tfn_matrix[i, j] = get_tfn(int(ratings[i, j]), self.scale)

        self._tfn[expert_id] = tfn_matrix

    def add_expert_tfn_matrix(
        self, 
        expert_id: int,
        tfn_matrix: np.ndarray
    ) -> None:
        """
        Add a pre-built TFN matrix directly (bypassing the scale).

        Parameters
        ----------
            expert_id : int
            tfn_matrix : ndarray, shape (m, n, 3)
        """
        tfn_matrix = np.asarray(tfn_matrix, dtype=float)
        if tfn_matrix.shape != (self.n_alternatives, self.n_criteria, 3):
            raise ValueError(
                f'Expected TFN matrix of shape '
                f'({self.n_alternatives}, {self.n_criteria}, 3), '
                f'got {tfn_matrix.shape}.'
            )
        self._tfn[expert_id] = tfn_matrix

    #  Retrieval                                                           
    @property
    def expert_ids(self) -> list:
        """Sorted list of expert IDs that have been added."""
        return sorted(self._tfn.keys())

    @property
    def n_experts(self) -> int:
        """Number of experts whose data has been added."""
        return len(self._tfn)

    def get_tfn_matrix(self, expert_id: int) -> np.ndarray:
        """
        Return the TFN decision matrix for one expert.

        Parameters
        ----------
            expert_id : int

        Returns
        -------
            ndarray, shape (m, n, 3)
        """
        if expert_id not in self._tfn:
            raise KeyError(f'No data for expert_id={expert_id}.')
        return self._tfn[expert_id].copy()

    def get_tfn_matrices(self) -> list:
        """
        Return a list of TFN matrices for all experts, sorted by expert_id.

        Returns
        -------
            list of ndarray, each of shape (m, n, 3)
        """
        return [self._tfn[eid].copy() for eid in self.expert_ids]

    def get_raw_matrix(self, expert_id: int) -> np.ndarray:
        """
        Return the raw (integer) rating matrix for one expert.

        Available only if the expert was added via add_expert_matrix().

        Parameters
        ----------
            expert_id : int

        Returns
        -------
            ndarray, shape (m, n), dtype int
        """
        if expert_id not in self._raw:
            raise KeyError(
                f'No raw (integer) data for expert_id={expert_id}. '
                'Raw data is only stored when using add_expert_matrix().'
            )
        return self._raw[expert_id].copy()

    #  Summary                                                             
    def __repr__(self) -> str:
        return (
            f'ExpertCollector('
            f'n_alternatives={self.n_alternatives}, '
            f'n_criteria={self.n_criteria}, '
            f'n_experts={self.n_experts})'
        )
