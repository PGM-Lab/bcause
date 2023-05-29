from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(points):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, projection="3d")

    vacuous = [list(zip([1, 0, 0], [0, 1, 0], [0, 0, 1]))]
    ax.add_collection3d(Poly3DCollection(vacuous, facecolors='gray', linewidths=1, alpha=0.6))

    xs = [[p[0] for p in points]]
    ys = [[p[1] for p in points]]
    zs = [[p[2] for p in points]]
    ax.scatter(xs, ys, zs, depthshade=False, c="red", marker="x", s=40)
    ax.view_init(30, 20)
    plt.show()
