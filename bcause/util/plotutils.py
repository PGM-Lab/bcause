from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def rgb_to_hex(rgb):
    """
    Convert an RGB color value to its hexadecimal representation.

    Parameters:
        rgb (tuple): A tuple containing the RGB color values.

    Returns:
        str: The color value in hexadecimal format.
    """

    r, g, b = rgb[:3]
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


def get_linear_colors(n, colormap_name='viridis'):
 # Create a colormap object
    colormap = plt.get_cmap(colormap_name)

    # Generate evenly spaced values from 0 to 1
    values = np.linspace(0, 1, n)

    # Get the corresponding colors from the colormap
    colors = [colormap(value) for value in values]

    # Convert colors to hexadecimal format
    hex_colors = [rgb_to_hex(color) for color in colors]

    return hex_colors


def get_xyz(points):
    # take first three dimensions of vectors in points and return them as x, y, z
    mat = np.array(points)
    return mat[:, 0], mat[:, 1], mat[:, 2]


def plot_3d(points, save_path = None, trajectories = None):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="3d")

    vacuous = [list(zip([1, 0, 0], [0, 1, 0], [0, 0, 1]))]
    ax.add_collection3d(Poly3DCollection(vacuous, facecolors='gray', linewidths=1, alpha=0.6))

    #xs = [[p[0] for p in points]]
    #ys = [[p[1] for p in points]]
    #zs = [[p[2] for p in points]]
    #xs, ys, zs = 
    ax.scatter(*get_xyz(points), depthshade=False, c="red", marker="x", s=40)
    if trajectories:
        colors = get_linear_colors(len(trajectories), 'viridis')
        for i, trajectory in enumerate(trajectories):
            ax.plot(*get_xyz(trajectory), c = colors[i])
    ax.view_init(30, 20)
    if save_path:
        ax.set_title(save_path)
        plt.savefig(save_path)
    else:
        plt.show(block=True)
