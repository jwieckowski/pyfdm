# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np
from ..TFN import TFN

def tfn_criteria_plot(criteria, ax=None):
    """
        Visualize Triangular Fuzzy Number (TFN) criteria weights on subplots.

        Parameters:
        - criteria (list): A list of lists, where each inner list represents the parameters (a, b, c) of a TFN criteria weight.
        - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

        Example:
        ```
        # Example Usage:
        criteria = [[0.2, 0.5, 0.8], [0.4, 0.7, 1.0], [0.1, 0.3, 0.6]]
        tfn_criteria_plot(criteria)
        ```

        The `criteria` parameter should be a list of lists, where each inner list contains three values representing the parameters (a, b, c) of a Triangular Fuzzy Number (TFN) criteria weight. The function generates subplots for each criteria weight, labeling them with 'Criteria C_i Weight' (where i is the index of the criteria weight in the input data).

        The number of rows and columns for subplots are determined based on the number of criteria weights. The maximum number of rows is set to 4, and additional columns are added if there are more weights.

        The function uses Matplotlib for plotting, and the resulting plot is displayed.
    """

    criteria_weights = [TFN(*crit) for crit in criteria]

    # Determine the number of rows and columns for subplots
    num_weights = len(criteria_weights)
    max_rows = num_weights if num_weights < 4 else 4
    num_cols = (num_weights - 1) // max_rows + 1

    if ax is None:
        fig, axs = plt.subplots(max_rows, num_cols, figsize=(12, 8))

    # Generate x values for the plot
    x_values = np.linspace(min(cw.a for cw in criteria_weights) - 0.1, max(cw.c for cw in criteria_weights) + 0.1, 1000)

    # Plot each criteria weight in subplots
    for i, cw in enumerate(criteria_weights):
        row_index = i % max_rows
        col_index = i // max_rows
        y_values = [cw.membership_function(x) for x in x_values]
        if num_cols == 1:
            axs[row_index].plot(x_values, y_values)
            axs[row_index].fill_between(x_values, 0, y_values, color='lightgray', alpha=0.7)
            axs[row_index].grid(True, linestyle='--', alpha=0.7)
            axs[row_index].axhline(0, color='black', linewidth=1)
            axs[row_index].set_ylabel('Membership Value $\mu$(x)')
            axs[row_index].set_title(f'Criteria $C_{{{i+1}}}$ Weight')
        elif num_cols > 1:
            axs[row_index, col_index].plot(x_values, y_values)
            axs[row_index, col_index].fill_between(x_values, 0, y_values, color='lightgray', alpha=0.7)
            axs[row_index, col_index].grid(True, linestyle='--', alpha=0.7)
            axs[row_index, col_index].axhline(0, color='black', linewidth=1)
            axs[row_index, col_index].set_ylabel('Membership Value $\mu$(x)')
            axs[row_index, col_index].set_title(f'Criteria $C_{{{i+1}}}$ Weight')

    # Adjust layout for better spacing
    plt.tight_layout()

    return axs
