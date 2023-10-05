'''
Auxiliary functions for MLE 
'''

import matplotlib.pyplot as plt

from bcause.util.plotutils import get_linear_colors


def convert_to_filename(values, file_extension=".png"):
    # Convert values to string representation
    value_strings = [str(value) for value in values]

    # Join the strings using underscore as a separator
    joined_string = "_".join(value_strings)

    # Remove any leading or trailing invalid characters
    valid_filename = "".join(c for c in joined_string if c.isalnum() or c in "._- ")

    valid_filename = valid_filename.replace('_', '__')
    valid_filename = valid_filename.replace('.', '_')

    # Append file extension
    filename = valid_filename + file_extension

    return filename


def plot_lkh_evol(trajectories, save_path = None):
    from dev.mle import negative_log_likelihood, inverse_transform_params
    from dev.optimization import py_x, fy, pu
    colors = get_linear_colors(len(trajectories), 'viridis')
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    max_iter = 0 # the number of iterations of the longest trajectory
    for i, trajectory in enumerate(trajectories):
        lkh = []
        max_iter = max(max_iter, len(trajectory))
        for params in trajectory:
            lkh.append(-negative_log_likelihood(inverse_transform_params(params), py_x, fy))
        ax.plot(range(len(lkh)), lkh, c = colors[i], marker = 'o', alpha = .5)
    ax.set_xlabel('iteration')
    ax.set_ylabel('likelihood')

    max_lkh = -negative_log_likelihood(inverse_transform_params(pu.values), py_x, fy)
    ax.plot((0, max_iter), (max_lkh, max_lkh), 'k--')
    if save_path:
        ax.set_title(save_path)
        plt.savefig(save_path)
    else:
        plt.show(block=True)