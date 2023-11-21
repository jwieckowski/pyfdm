# Copyright (c) 2023 Jakub Więckowski

import numpy as np
import matplotlib.pyplot as plt
# from ..TFN import TFN
from pyfdm.TFN import TFN

def multiple_tfn_plot(data, title=None, ax=None):
    """
        Visualize multiple Triangular Fuzzy Numbers (TFNs) on a single plot.

        Parameters:
        - data (list): A list of lists, where each inner list represents the parameters (a, b, c) of a TFN.
        - title (str, optional): The title of the plot. If not provided, the default title is 'Triangular Fuzzy Numbers'.
        - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

        Example:
        ```
        # Example Usage:
        data = [[2, 3, 6], [3, 5, 7], [6, 8, 9]]
        multiple_tfn_plot(data, title='Example TFN Plot')
        ```

        The `data` parameter should be a list of lists, where each inner list contains three values representing the parameters (a, b, c) of a Triangular Fuzzy Number (TFN). The function generates a plot with the TFNs, labeling each curve with 'TFN i' (where i is the index of the TFN in the input data). The areas under the curves are filled with light gray for better visualization.

        The title of the plot can be specified using the optional `title` parameter. If no title is provided, the default title is 'Triangular Fuzzy Numbers'.

        The function uses Matplotlib for plotting, and the resulting plot is displayed.
    """
    if ax is None:
        ax = plt.gca()

    data = np.array(data)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            'TFN matrix elements should all have length of 3')

    tfns = []
    for item in data:
        tfns.append(TFN(*item))

    # Generate x values for the plot
    x_values = np.linspace(min([tfn.a for tfn in tfns]) - 1, max([tfn.c for tfn in tfns]) + 1, 1000)

    # Calculate y values for each TFN
    y_values = []
    for tfn in tfns:
        y_values.append([tfn.membership_function(x) for x in x_values])

    # Plot the triangular fuzzy numbers
    for idx, y in enumerate(y_values):
        ax.plot(x_values, y, label=f'TFN {idx+1}', linewidth=2)

        # Fill the areas under the curves
        ax.fill_between(x_values, 0, y, color='lightgray', alpha=0.7)

    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)

    ax.set_ylabel('Membership Value $\mu$(x)')
    title = title if title is not None else 'Triangular Fuzzy Numbers' 
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax
