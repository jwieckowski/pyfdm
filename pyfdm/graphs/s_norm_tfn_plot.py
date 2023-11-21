# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np
from ..TFN import TFN

def s_norm_tfn_plot(data, title=None, ax=None):
    """
        Visualize the S-Norm operation on a set of Triangular Fuzzy Numbers (TFNs).

        Parameters:
        - data (list): A list of lists, where each inner list represents the parameters (a, b, c) of a TFN.
        - title (str, optional): The title of the plot. If not provided, the default title is 'S-norm Operation'.
        - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

        Example:
        ```
        # Example Usage:
        data = [[2, 3, 6], [3, 5, 7], [6, 8, 9]]
        s_norm_tfn_plot(data, title='S-Norm Visualization')
        ```

        This function generates a plot illustrating the S-Norm operation on a set of Triangular Fuzzy Numbers (TFNs). The TFNs are defined by parameters a, b, and c, provided in the input data. Each individual TFN is plotted, and the result of the S-Norm operation is visualized with a dashed line labeled 'S-norm'. The vertical dashed lines represent the TFN parameters (a, b, c).

        The title of the plot can be specified using the optional `title` parameter. If no title is provided, the default title is 'S-norm Operation'.

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

    # Plot individual TFNs
    for idx, y in enumerate(y_values):
        ax.plot(x_values, y, label=f'TFN {idx+1}', linewidth=2)

    # Plot S-norm result
    y_values = np.array(y_values)
    y_values_s_norm = np.max(y_values, axis=0)
    ax.plot(x_values, y_values_s_norm, label='S-norm', linestyle='--', linewidth=4, color='green')

    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)

    ax.set_ylabel('Membership Value $\mu$(x)')
    title = title if title is not None else 'S-norm Operation' 
    ax.set_title(title)
    ax.legend()

    return ax