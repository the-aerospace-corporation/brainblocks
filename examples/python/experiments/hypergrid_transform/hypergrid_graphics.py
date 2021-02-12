
# brainblocks - gnome similarity function
from brainblocks.metrics import *

# shapely - creating polygons for grid visualization
from shapely.ops import linemerge, polygonize_full, polygonize, unary_union, cascaded_union
from shapely.geometry import *

# matplotlib - drawing the plot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, NoNorm
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib import ticker

# numpy/scipy - math and geometry
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

#from pprint import PrettyPrinter


def clamp_point(point, minx=-1, miny=-1, maxx=1, maxy=1):
    px = point[0]
    py = point[1]

    if px > maxx:
        newx = maxx
    elif px < minx:
        newx = minx
    else:
        newx = px

    if py > maxy:
        newy = maxy
    elif py < miny:
        newy = miny
    else:
        newy = py

    return newx, newy


def compute_distance_to_plot_boundary(direction_vec, bbox=((1.0, 1.0), (-1.0, -1.0)), origin=np.array([0.0, 0.0])):
    # FIXME: assuming that origin is centered with bounding box
    # origin = np.array([0., 0.])

    x = direction_vec[0]
    y = direction_vec[1]

    # range of +/- np.pi
    angle = np.arctan2(y, x)

    # distance from origin to x_pos boundary
    x_pos = np.fabs(bbox[0][0] - origin[0])

    # distance from origin to x_neg boundary
    x_neg = np.fabs(bbox[1][0] - origin[0])

    # distance from origin to y_pos boundary
    y_pos = np.fabs(bbox[0][1] - origin[1])

    # distance from origin to y_neg boundary
    y_neg = np.fabs(bbox[1][1] - origin[1])

    # np.tan(angle) = y_leg / x_leg

    # upper direction
    if angle >= np.pi / 4 and angle < 3 * np.pi / 4:
        y_leg = y_pos
        x_leg = y_leg / np.tan(angle)
    # right direction
    elif angle < np.pi / 4 and angle >= -np.pi / 4:
        x_leg = x_pos
        y_leg = x_leg * np.tan(angle)
    # down direction
    elif angle < -np.pi / 4 and angle >= -3 * np.pi / 4:
        y_leg = y_neg
        x_leg = y_leg / np.tan(angle)
    # left direction
    # elif angle >= 3 * np.pi / 4 or angle <= -3 * np.pi / 4:
    else:
        x_leg = x_neg
        y_leg = x_leg * np.tan(angle)

    distance = np.sqrt(x_leg * x_leg + y_leg * y_leg)

    return distance


def draw_gnomes(ax, X_gnomes, num_bins, num_grids=1, fontsize=16, colors=None, annot=None):
    if annot is None:
        annot = np.empty(shape=X_gnomes.shape, dtype=np.object)
        for i in range(num_bins):
            for j in range(num_bins):
                annot[i, j] = "%d,%d" % (i, j)

    if colors is None:
        colors_pal = sns.color_palette("deep", n_colors=num_grids)
        # colors for the heatmap representation
        colors = [np.array(colorConverter.to_rgba("white"))]
        for k in range(num_grids):
            curr_color = np.array(colorConverter.to_rgba(colors_pal[k]))
            colors.append(curr_color)

    gnome_plot = sns.heatmap(X_gnomes, ax=ax, square=True, linewidths=0.05, linecolor='k', cbar=False,
                             cmap=colors, annot=annot, fmt='',
                             annot_kws={'fontsize': fontsize, 'color': 'k', 'alpha': 0.4},
                             clip_on=False)
    # X-Axis bin labels
    gnome_plot.set_xticklabels([])
    gnome_plot.xaxis.set_major_locator(ticker.IndexLocator(num_bins, num_bins / 2))
    # gnome_plot.set_xticklabels([k for k in range(num_bins)], fontdict={'fontsize': 12})

    # Y-Axis hypergrid angle labels
    gnome_plot.yaxis.set_major_locator(ticker.IndexLocator(num_bins, num_bins / 2))
    gnome_plot.set_yticklabels([])
    gnome_plot.set_aspect('equal')


def draw_grid(ax, grid_bases, num_bins=4):
    grid_range = 3

    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    r90 = np.matrix(((c, -s), (s, c)))

    xrange = np.arange(-grid_range, grid_range + 1 / num_bins, 1 / num_bins)
    yrange = np.arange(-grid_range, grid_range + 1 / num_bins, 1 / num_bins)

    line_list = []
    for grid_i in range(grid_bases.shape[0]):

        lattice_bases = grid_bases[grid_i]
        ortho_bases = np.array(np.matmul(lattice_bases, r90))

        # x lines
        for x in xrange:
            # Input Space Coordinates
            grid_origin = lattice_bases[0] * (x - 1 / (2 * num_bins))

            grid_0 = grid_origin + ortho_bases[0] * grid_range
            grid_1 = grid_origin - ortho_bases[0] * grid_range

            grid_0 = tuple(grid_0)
            grid_1 = tuple(grid_1)

            line_list.append((grid_0, grid_1))

        for y in yrange:
            # Input Space Coordinates
            # grid_origin = np.matmul(lattice_bases, offset_p)
            grid_origin = lattice_bases[1] * (y - 1 / (2 * num_bins))

            grid_0 = grid_origin + ortho_bases[1] * grid_range
            grid_1 = grid_origin - ortho_bases[1] * grid_range
            grid_0 = tuple(grid_0)
            grid_1 = tuple(grid_1)

            line_list.append((grid_0, grid_1))

    # shapely
    multi_lines = MultiLineString(line_list)
    merged = linemerge(list(multi_lines))
    borders = unary_union(merged)
    polygons = polygonize(borders)

    # matplotlib
    for p in polygons:
        ax.plot(*p.exterior.xy, color='k', alpha=1.0, linewidth=0.4)


def draw_voronoi(ax, grid_bases, num_bins=4):
    x_basis = np.array([np.sqrt(3.0) / 3.0, 0.0])
    y_basis = np.array([-1.0 / 3.0, 2.0 / 3.0])

    hex_bases = np.array([[
        [np.sqrt(3.0) / 3.0, 0.0],
        [-1.0 / 3.0, 2.0 / 3.0]
    ]])

    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R90 = np.matrix(((c, -s), (s, c)))
    ortho_bases = np.asarray([np.matmul(grid_bases[0], R90.T)])

    # pointy orientation dimensions
    # size distance from center to corner
    # w = sqrt(3) * size
    # h = 2 * size
    # distance between adjacent hexagon centers is w
    # vertical distance between adjacent hexagon centers is h*3/4

    print("ortho_bases:", ortho_bases.shape, ortho_bases)

    # build lattice of points of the grid
    xx, yy = np.meshgrid(np.arange(-12, 12 + 1, 1),
                         np.arange(-12, 12 + 1, 1))
    XY_lattice = np.c_[xx.ravel(), yy.ravel()]
    XY_lattice = XY_lattice / num_bins

    print("lattice indices:")
    print(XY_lattice.shape, XY_lattice[:9, :])

    print("grid_bases:", grid_bases[0].shape, grid_bases[0])

    inv_bases = np.linalg.inv(grid_bases[0])
    print("inv_bases:", inv_bases.shape, inv_bases)

    disp_vec = np.array([1, 1])

    print("feature point:", disp_vec.shape, disp_vec)

    # matrix multiply @ to get projection to subspaces
    # proj_vec: (num_grids, num_subspace_dims, 1)
    # proj_vec: (4, 2, 1)
    proj_vec = np.matmul(grid_bases[0], disp_vec)
    print("subspace point:", proj_vec.shape, proj_vec)

    inv_scaled_bases = np.linalg.inv(grid_bases[0])

    lattice_bases = inv_scaled_bases
    print("lattice_bases:", lattice_bases.shape, lattice_bases)

    feature_point = np.matmul(ortho_bases[0], proj_vec)
    print("feature point:", feature_point.shape, feature_point)
    feature_point = np.matmul(proj_vec, ortho_bases[0])
    print("feature point:", feature_point.shape, feature_point)
    feature_point = np.matmul(inv_bases, proj_vec)
    print("feature point:", feature_point.shape, feature_point)

    point_list = []
    for p in XY_lattice:
        point_list.append(np.matmul(lattice_bases, p))

    lattice_points = np.array(point_list)

    print("lattice_points")
    print(lattice_points.shape, lattice_points[:9, :])

    lattice_points = lattice_points

    vor = Voronoi(lattice_points)
    voronoi_plot_2d(vor, ax=ax, line_alpha=0.5, line_width=0.5, show_points=False, show_vertices=False, line_colors='k')


def draw_similarity(ax, hgt, X_gnomes, color=None):
    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                         np.arange(y_min, y_max + h, h))
    # print("mesh x/y size:", xx.shape, yy.shape)

    # mesh put together with x/y coordinates
    XY_mesh = np.c_[xx.ravel(), yy.ravel()]
    # print("evaluation meshgrid shape")
    # print(XY_mesh.shape)

    XY_gnomes = hgt.transform(XY_mesh)

    XY_gnomes = XY_gnomes.reshape(xx.shape[0], xx.shape[1], -1)

    heat_colors = sns.color_palette("Reds", n_colors=10)
    for i, clr in enumerate(heat_colors):
        temp_color = list(heat_colors[i])
        if len(temp_color) == 3:
            temp_color += [1.0, ]
        temp_color[3] = i / 10
        heat_colors[i] = temp_color

    gray_colors = sns.color_palette("Greys", n_colors=10)
    for i, clr in enumerate(gray_colors):
        temp_color = list(gray_colors[i])
        if len(temp_color) == 3:
            temp_color += [1.0, ]
        temp_color[3] = i / 10
        gray_colors[i] = temp_color

    # print("similarity contour colors")
    # print(heat_colors)
    no_norm = NoNorm(vmin=0, vmax=1.0, clip=True)
    heat_cmap = ListedColormap(heat_colors)
    gray_cmap = ListedColormap(gray_colors)

    mesh_gnome_result = gnome_similarity(X_gnomes.reshape(1, -1),
                                         XY_gnomes.reshape(-1, XY_gnomes.shape[2]))
    mesh_gnome_result = mesh_gnome_result.reshape(XY_gnomes.shape[0:2])

    contour_set = ax.contourf(xx, yy, mesh_gnome_result, levels=10, norm=no_norm, cmap=gray_cmap)
    # contour_set = ax.contourf(xx, yy, mesh_gnome_result, levels=10, norm=no_norm, cmap=heat_cmap, nchunk=0,
    #                          antialiased=False)

    return contour_set


def draw_bases(ax, hgt, colors=None):
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    scaled_bases = np.array([np.multiply(hgt.subspace_vectors[k], hgt.subspace_periods[k].reshape(-1, 1)) for k in
                             range(hgt.subspace_vectors.shape[0])])

    num_grids = hgt.num_grids
    num_subspace_dims = hgt.num_subspace_dims
    num_bins = hgt.num_bins

    for grid_i in range(num_grids):
        color = colors[grid_i]
        V = scaled_bases[grid_i]

        # origin = np.array([-1.5, 1.5])  # origin point
        origin = np.array([0, 0])  # origin point

        ax.quiver(origin[0], origin[1], V[:, 0], V[:, 1], color=color, alpha=0.5, clip_on=False, angles='xy',
                  scale_units='xy', scale=1)

        print("V:", V)

        for dim_i in range(num_subspace_dims):
            vec = V[dim_i]
            print("vec:", vec)
            # period = np.linalg.norm(vec)
            offset = -vec / (num_bins * 2)
            print("offset:", offset)
            print("origin:", origin)
            print("origin+offset:", origin + offset)

            points = [origin + offset]

            for bound_i in range(1, num_bins + 1):
                points.append(origin + offset + vec * bound_i / num_bins)

            points = np.array(points)

            print("points:", points)
            # ax.scatter(points[:, 0], points[:, 1], color='k', alpha=0.5, clip_on=False)
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, marker='o', clip_on=False)


def draw_lines(ax, hgt, ortho_grid_vecs, ortho_grid_angles, points, artists=None, colors=None):
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    origin_x, origin_y = hgt.origin

    if artists is None:
        point_line_artists = []
    else:
        point_line_artists = artists

    for iter_index, pnt in enumerate(points):

        # first grid angles only
        ortho_angles = ortho_grid_angles[0]
        ortho_vecs = ortho_grid_vecs[0]

        for dim_i in range(hgt.num_subspace_dims):
            ortho_angle = ortho_angles[dim_i]
            ortho_vec = ortho_vecs[dim_i]

            # projected distance to origin from sample point
            point_to_origin_dist = compute_distance_to_plot_boundary(ortho_vec, bbox=((2.0, 2.0), (-2.0, -2.0)))

            point_to_origin_dist = 6.0

            # line drawn from sample point to boundary of grid visual
            line_x = pnt[0] + np.cos(np.radians(ortho_angle)) * point_to_origin_dist
            line_y = pnt[1] + np.sin(np.radians(ortho_angle)) * point_to_origin_dist
            if artists is None:
                lines = ax.plot([pnt[0], line_x], [pnt[1], line_y], clip_on=False, color=colors[iter_index],
                                alpha=0.3)  # , zorder=6)

                point_line_artists.append(lines[0])
            else:
                point_line_artists[iter_index + dim_i].set_data([pnt[0], line_x], [pnt[1], line_y])
                point_line_artists[iter_index + dim_i].set_color(colors[iter_index])
                point_line_artists[iter_index + dim_i].set_alpha(0.3)

    return point_line_artists


def set_hgt_artists(ax, artists, hgt, ortho_grid_vecs, ortho_grid_angles, points, colors=None):
    # set_hgt_artists(ax0, ax_artists, hgt, points)

    # artists.shape = (hgt.num_grids, hgt.num_subspace_dims, hgt.num_bins, hgt.num_bins)

    num_grids = hgt.num_grids
    num_subspace_dims = hgt.num_subspace_dims

    bits_tensor = hgt.transform(points)
    # print("bits_tensor:", bits_tensor.shape)
    # print("point_bits:", point_bits.shape)
    # print("grid_bits:", grid_bits.shape)

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    for grid_i in range(num_grids):
        color = colors[grid_i]
        # print("GRID", grid_i)
        # print()
        point_bits = bits_tensor[0]
        grid_bits = point_bits[grid_i]

        if num_subspace_dims == 1:
            pass
        elif num_subspace_dims == 2:
            pass

            for dim_i in range(hgt.num_subspace_dims):

                if dim_i == 0:
                    bits_view = grid_bits
                else:
                    bits_view = grid_bits.T

                num_x_bins = hgt.num_bins
                num_y_bins = hgt.num_bins

                # period_x = periods[grid_i, dim_i % hgt.num_subspace_dims]
                # period_y = periods[grid_i, (dim_i + 1) % hgt.num_subspace_dims]

                for i in range(num_x_bins):
                    for j in range(num_y_bins):
                        is_occupied = bits_view[i, j]

                        rect_artist, text_artist = artists[grid_i][dim_i][i][j]
                        if is_occupied:
                            rect_artist.set_edgecolor('k')
                            rect_artist.set_facecolor(color)
                        else:
                            rect_artist.set_edgecolor('k')
                            rect_artist.set_facecolor('none')
        else:
            pass


def build_hgt_artists(ax=None, hgt=None, colors=None):
    # extra padding distance from plot boundary for rendering grid artist
    # padding = 0.5
    padding = 0.1

    # orient the grid visual orthogonal to its vector direction of sensitivity
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R90 = np.matrix(((c, -s), (s, c)))
    # print("Rotation Matrix by 90:", R90.shape, R90)
    # print()

    # returned artists data structure, nested list of grids and subspace dimensions
    # grid artists, artists[grid_i][subspace_i] = np.empty([hgt.num_bins, hgt_bins], dtype=np.object)
    # elements of numpy array contain artist tuples of (rect, internal_text)
    artists = np.empty([hgt.num_grids, hgt.num_subspace_dims, hgt.num_bins, hgt.num_bins], dtype=np.object)

    num_input_dims = hgt.num_features
    num_subspace_dims = hgt.num_subspace_dims
    periods = hgt.subspace_periods
    basis_vectors = hgt.subspace_vectors
    num_bins = hgt.num_bins
    num_acts = hgt.num_acts
    num_grids = hgt.num_grids
    max_period = hgt.max_period
    min_period = hgt.min_period
    origin = hgt.origin

    ortho_grid_vecs = []
    ortho_grid_angles = []

    print("params")
    print(num_input_dims, num_subspace_dims, num_bins, num_acts, num_grids)
    print("num_grids=", num_grids)

    # selection of subspace bases forms a lattice
    # can we have num_subspace_dims greater than num_input_dims?

    for grid_i in range(num_grids):
        print("GRID", grid_i, num_grids)
        print()

        if num_subspace_dims == 1:
            # if 1D, create 1D grid
            # vertically increment so further grids can be stacked visually

            # option #1:  create aligned 1D grids

            basis_vector0 = basis_vectors[grid_i][0]
            # print("basis vector:")
            # print(basis_vector0.shape)
            # print(basis_vector0)
            # print()

            # compute angles of basis vectors
            basis_angle = np.degrees(np.arctan2(basis_vector0[1], basis_vector0[0]))
            # print("basis_angle:")
            # print(basis_angle)

            # get orthogonal direction to grid's sensitive direction
            ortho_vec = np.ravel(np.matmul(basis_vector0, R90.T))
            # print("ortho_vec:")
            # print(ortho_vec.shape)
            # print(ortho_vec)

            # get displacement in orthgonal direction
            # displacement_vector = displacement * ortho_vec
            # print("displacement_vector:")
            # print(displacement_vector.shape)
            # print(displacement_vector)

            # if the origin has moved, displace the visuals from this origin
            # rotation_point = hgt.origin
            # origin_x, origin_y = Affine2D().transform(
            #    [rotation_point[0] + displacement_vector[0], rotation_point[1] + displacement_vector[1]])
            # print("plot origin_x, origin_y:")
            # print(origin_x, origin_y)

        elif num_subspace_dims == 2:
            # if 2D, create 2D grid for each axis of subspace
            # disallow stacking of aligned grids for now

            bases = basis_vectors[grid_i]

            basis_angles = np.degrees(np.unwrap(np.arctan2(bases[:, 1], bases[:, 0])))

            # get orthogonal direction to grid's sensitive direction
            ortho_vecs = np.asarray(np.matmul(bases, R90.T))
            ortho_angles = np.degrees(np.unwrap(np.arctan2(ortho_vecs[:, 1], ortho_vecs[:, 0])))

            ortho_grid_vecs.append(ortho_vecs)
            ortho_grid_angles.append(ortho_angles)

            displacements = np.apply_along_axis(compute_distance_to_plot_boundary, 1,
                                                ortho_vecs) + padding + (grid_i % 2) * 1.4

            # get displacement in orthgonal direction
            displacement_vectors = np.multiply(ortho_vecs, displacements.reshape(-1, 1))

            # if the origin has moved, displace the visuals from this origin
            rotation_point = hgt.origin

            # point at which plotting the grid starts
            plot_origins = displacement_vectors + rotation_point

            for dim_i in range(hgt.num_subspace_dims):
                origin_x = plot_origins[dim_i % hgt.num_subspace_dims][0]
                origin_y = plot_origins[dim_i % hgt.num_subspace_dims][1]

                basis_angle = basis_angles[dim_i % hgt.num_subspace_dims]
                ortho_angle = ortho_angles[dim_i % hgt.num_subspace_dims]

                # data space coordinates of visual to find new point after rotated
                fixed_point_rotation = Affine2D().rotate_deg_around(origin_x, origin_y, basis_angle)

                num_x_bins = num_bins
                num_y_bins = num_bins

                period_x = periods[grid_i, dim_i % hgt.num_subspace_dims]
                period_y = periods[grid_i, (dim_i + 1) % hgt.num_subspace_dims]

                # bin period
                bin_interval_x = period_x / num_bins
                bin_interval_y = period_y / num_bins

                # position of bottom left corner of the first bin rectangle of this grid
                grid_x = origin_x - bin_interval_x / 2
                grid_y = origin_y

                for i in range(num_x_bins):
                    for j in range(num_y_bins):
                        # bottom left corner of current bin rectangle
                        bin_boundary_x = grid_x + bin_interval_x * i
                        if dim_i == 0:
                            bin_boundary_y = grid_y + bin_interval_y * j
                        else:
                            bin_boundary_y = grid_y - bin_interval_y * j + bin_interval_y * (num_y_bins - 1)

                        # bottom left corner after its been rotated
                        corner_pos = fixed_point_rotation.transform([bin_boundary_x, bin_boundary_y])

                        if dim_i == 0:
                            text_str = "%d,%d" % (i, j)
                        else:
                            text_str = "%d,%d" % (j, i)

                        # create rectangle with text inside
                        rect, internal_text = create_text_rect_artists(ax, corner_pos[0], corner_pos[1], bin_interval_x,
                                                                       bin_interval_y,
                                                                       angle=basis_angle,
                                                                       text_str=text_str,
                                                                       fontsize=10,
                                                                       aligned_text=False)

                        artists[grid_i, dim_i, i, j] = (rect, internal_text)
                        # print("setting artist:", grid_i, dim_i, i, j)


        elif num_subspace_dims > 2:
            # if greater than 2 dimensions, raise error, no way to visualize
            raise Exception("Error: num_subspace_dims > 2, can only visualize up to 2 dimensions")

    return artists, ortho_grid_vecs, ortho_grid_angles


def create_text_rect_artists(ax, box_x, box_y, box_width, box_height, angle=0, linewidth=1.5, edgecolor='k', fontsize=8,
                             facecolor='none', text_str=None, aligned_text=False, alpha=1.0):
    text_v_offset = -0.01

    # data space coordinates to find new point after rotated
    fixed_point_rotation = Affine2D().rotate_deg_around(box_x, box_y, angle)

    # add rectangle at corner position and rotate by angle
    rect = patches.Rectangle((box_x, box_y), box_width, box_height, angle=angle, linewidth=linewidth,
                             edgecolor=edgecolor,
                             facecolor=facecolor, clip_on=False, alpha=alpha)
    rect_patch = ax.add_patch(rect)

    # put angle within +180/-180
    normalized_angle = angle
    if angle > 0:
        while normalized_angle > 180:
            normalized_angle -= 360
    elif angle < 0:
        while normalized_angle < -180:
            normalized_angle += 360

    # angle the textbox nicely
    if aligned_text:
        text_angle = normalized_angle
        if abs(text_angle) > 90:
            text_angle += 180
    else:
        text_angle = 0

    # space the text box nicely so it fits no matter orientation
    # nice centering depends on orientation of text in the figure
    if aligned_text:

        # upper quadrants and bottom quadrants have different text orientation and adjustment
        if abs(normalized_angle) > 90:
            # upper quadrant adjustment
            rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5 - text_v_offset]
        else:
            # bottom quadrant adjustment
            rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5 + text_v_offset]
    else:
        rect_center_pos = [box_x + box_width / 2, box_y + box_height * 0.5]

    # rotate around rectangle corner
    text_pos = fixed_point_rotation.transform(rect_center_pos)

    # unaligned text uses standard vertical offset in axes frame
    if not aligned_text:
        text_pos[1] += text_v_offset

    # add text box to center of rectangle
    internal_text = ax.text(text_pos[0], text_pos[1], text_str, rotation=text_angle, rotation_mode='anchor',
                            fontsize=fontsize, va='center', ha='center', clip_on=False, alpha=alpha)

    return rect_patch, internal_text


