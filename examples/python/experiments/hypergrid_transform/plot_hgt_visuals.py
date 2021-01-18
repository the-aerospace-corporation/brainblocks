import glob
import os
import seaborn as sns

# brainblocks
from brainblocks.tools import HyperGridTransform
from brainblocks.datasets.time_series import lemiscate

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, NoNorm
from matplotlib import ticker

# illustrations
from hypergrid_graphics import set_hgt_artists, build_hgt_artists, draw_bases, draw_grid, draw_gnomes, \
    draw_lines, draw_similarity, draw_voronoi

import numpy as np


def run(filename_root="visual_test_frame_1", output_dir="out"):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create filename path
    filename_str = output_dir + "/" + filename_root + "_%05u.png"

    # remove old files if exist
    results = glob.glob(output_dir + "/" + filename_root + "_*")
    for filename in results:
        os.remove(filename)

    fig = plt.figure(num=1, figsize=(8, 8))  # , constrained_layout=True)
    # fig = plt.figure(num=1, figsize=(12, 8))  # , constrained_layout=True)
    # fig, axes = plt.subplots(1, 2, num=1, gridspec_kw={"width_ratios": [0.6, 0.5]})
    fig, axes = plt.subplots(1, 1, num=1)  # , gridspec_kw={"width_ratios": [0.6, 0.5]})

    # axes for hypergrid visual
    # ax0 = axes[0]
    ax0 = axes

    fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.7)  # , wspace=0.2)
    # fig.subplots_adjust(left=0.3, right=1.0, bottom=0.1, top=0.6)  # , wspace=0.2)
    # fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.2)
    # fig.subplots_adjust(wspace=0.8)

    # points for initializing HyperGrid Transform
    points = np.array([[0.0, 0.0]])

    num_input_dims = 2
    num_subspace_dims = 2
    num_grids = 3

    # hex_bases = np.array([[
    #   [np.sqrt(3.0) / 3.0, 0.0],
    #   [-1.0 / 3.0, 2.0 / 3.0]
    # ]])

    hex_bases = np.array([[
        [1.0, 0.0],
        [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)]
    ]])
    # hex_bases = np.array([[
    #    [1.0, 0.0],
    #    [np.cos(3 * np.pi / 4), np.sin(3 * np.pi / 4)]
    # ]])

    hex_bases = np.array([
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        [
            [1.0, 0.0],
            [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)]
        ],
        [
            [np.sqrt(2) / 2., np.sqrt(2) / 2],
            [-np.sqrt(2) / 2., np.sqrt(2) / 2]
        ]
    ])

    # get unit basis and periods from arbitrary vectors
    norms = np.apply_along_axis(np.linalg.norm, 2, hex_bases)
    custom_periods = norms
    custom_bases = np.array([np.divide(hex_bases[k], norms[k].reshape(-1, 1)) for k in range(hex_bases.shape[0])])

    # grid for visualizing
    hgt = HyperGridTransform(num_grids=num_grids, num_bins=4, num_subspace_dims=2, set_bases=custom_bases,
                             set_periods=custom_periods).fit(points)
    # hgt = HyperGridTransform(num_grids=num_grids, num_bins=4, num_subspace_dims=2).fit(points)

    # color palette to distinguish the grids
    colors = sns.color_palette("deep", n_colors=num_grids)

    # colors for the gnome heatmap representation
    colors2 = [np.array(colorConverter.to_rgba("white"))]
    for k in range(hgt.num_grids):
        curr_color = np.array(colorConverter.to_rgba(colors[k]))
        colors2.append(curr_color)

    bits_tensor = hgt.transform(points)

    scaled_bases = np.array([np.multiply(hgt.subspace_vectors[k], hgt.subspace_periods[k].reshape(-1, 1)) for k in
                             range(hgt.subspace_vectors.shape[0])])

    # draw grid with selected grid bases
    # draw_voronoi(ax0, scaled_bases)
    draw_grid(ax0, scaled_bases)
    draw_bases(ax0, hgt)

    ax0.set_aspect('equal')
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax0.tick_params(labelsize=8)
    tick_locs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    ax0.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
    ax0.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))

    # do visualization with given points
    # ax_artists, ortho_grid_vecs, ortho_grid_angles = build_hgt_artists(ax=ax0, hgt=hgt)

    # pp = PrettyPrinter()
    # pp.pprint(ax_artists)
    # [hgt.num_grids, hgt.num_subspace_dims, hgt.num_bins, hgt.num_bins]

    # print(ax_artists.shape)

    sns.set()

    # trajectory for moving point
    xs, ys = lemiscate(n=100, alpha=0.5)
    ys = 3 * ys
    num_points = len(xs)

    lines = None
    paths = None
    contour_set = None

    for point_index in range(num_points):

        points = [(xs[point_index], ys[point_index])]

        if not paths is None:
            paths.remove()

        paths = ax0.scatter(xs[point_index], ys[point_index], edgecolor='k', color='w', marker='o', s=60, zorder=100)

        # set_hgt_artists(ax0, ax_artists, hgt, ortho_grid_vecs, ortho_grid_angles, points, colors=colors)
        # lines = draw_lines(ax0, hgt, ortho_grid_vecs, ortho_grid_angles, points, artists=lines, colors=colors)

        X_gnomes = hgt.transform(points)  # assume one point
        # remove excess axis for samples
        X_gnomes = X_gnomes.reshape(hgt.num_grids * hgt.num_bins, hgt.num_bins).astype(int)

        if not contour_set is None:
            for coll in contour_set.collections:
                coll.remove()

        contour_set = draw_similarity(ax0, hgt, X_gnomes)

        indices = np.where(X_gnomes == 1)

        X_gnomes[indices] = indices[0] + 1

        # ax1 = axes[1]
        # ax1.clear()
        # draw_gnomes(ax1, X_gnomes, hgt.num_bins, num_grids=hgt.num_grids, colors=colors2)

        print("saving %s" % (filename_str % point_index))
        plt.savefig(filename_str % point_index, bbox_inches='tight')

    plt.clf()


# plot fading line

if __name__ == "__main__":
    run()
