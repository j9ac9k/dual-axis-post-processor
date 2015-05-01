# TODO Trim Domain such that there isn't a large
# TODO Generate timestamp for both encoder reading and power reading
# Using the known deltaT and the known velocity, recompute a more precise
# position that the sensor reading was taken at.

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import parser
from itertools import islice
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import style
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import savefig
import colorPicker
style.use('ggplot')
# insert after each print statement
# import sy_s
# sy_s.stdout.flush()




def process(args):
    # close all currently open plots
    plt.close("all")
    if args['simulated_data']:
        xi, yi, zi, args['scan_name'], args['pixel_pitch'], fault = parse_sim_data(args['filename'])
        if fault == 'pixel_pitch_fault':
            return fault
    else:
        x, y, z = parse_row_raw_data(args['filename'])
        xi, yi, zi = grid_interpolation(x, y, z, args['pixel_pitch'])
    xi, yi, x_offset, y_offset, long_axis_power, short_axis_power = center_origin(xi, yi, zi)

    # Setting The ColorMap
    cmap = colorPicker.retrieve_colormaps(args['colormap'], args['colormap_reverse'])[0]


    # ==============================================================================
    #     #determining grid-line spacing
    # ==============================================================================

    x_tick_boundary = int(max(abs(np.min(xi)), np.max(xi)))
    y_tick_boundary = int(max(abs(np.min(yi)), np.max(yi)))

    x_ticks = np.arange(0, x_tick_boundary, args['grid_resolution'])
    x_ticks = np.unique(np.array([-np.flipud(x_ticks), x_ticks]).flatten())

    y_ticks = np.arange(0, y_tick_boundary, args['grid_resolution'])
    y_ticks = np.unique(np.array([-np.flipud(y_ticks), y_ticks]).flatten())
    # draw rectangle
    rectangle = Rectangle((-args['target_width']/2,
                           -args['target_height']/2),
                          args['target_width'],
                          args['target_height'],
                          clip_on=False,
                          linewidth=3,
                          fill=False,
                          edgecolor='black',
                          ls='dashed')
    # ==============================================================================
    #     #calculate uniformity as a function of box size ratio
    # ==============================================================================
    if args['uniformity_vs_box_size_ratio_plot']:
        box_size_ratio = np.arange(-0.10, .330, 1 / (args['pixel_pitch'] * max(args['target_width'],
                                                                               args['target_height'])))
        uniformity = []
        for i in box_size_ratio:
            uniformity.append(calc_uniformity(i, args, xi, yi, zi)[0]*100)
        fig = plt.figure(num="Uniformity vs. Box Ratio (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Uniformity vs. Box Ratio Plot')
        ax.set_xlabel('Box Size Ratio')
        ax.set_ylabel('Uniformity (%)')
        ax.fill_between(box_size_ratio+1, uniformity, 0, color=colors.rgb2hex(cmap(0.5)), alpha=0.3)
        ax.plot(box_size_ratio+1, uniformity, color=colors.rgb2hex(cmap(0.5)))
        plt.axis([np.min(box_size_ratio)+1, np.max(box_size_ratio)+1, 0, np.max(uniformity)*1.1])
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " uniformity vs box ratio plot", dpi=200)

    # ==============================================================================
    #     #Plot Basic Heat Map
    # ==============================================================================
    if args['heat_map']:
        fig = plt.figure(num="Heat Map (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Heat Map')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        #plt.pcolormesh(xi, yi, zi, cmap=cmap, shading='gouraud', alpha=1.0)
        v = np.linspace(0, 1, 256+1, endpoint=True)
        plt.contourf(xi, yi, zi, v, cmap=cmap, shading='gouraud')
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        v = np.linspace(0, 1, 10+1, endpoint=True)
        plt.colorbar(cax=cax, ticks=v)
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " Heat Map", dpi=200)

    # ==============================================================================
    #     #Contour View
    # ==============================================================================
    if args['contour_plot']:
        fig = plt.figure(num="Contour Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Contour Plot')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        v = np.linspace(0, 1, args['contour_resolution']+1, endpoint=True)
        plt.contour(xi, yi, zi, v, linewidths=2, cmap=cmap)
        plt.clim(0, 1)
        ax.add_patch(rectangle)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(cax=cax, ticks=v)
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " contour plot", dpi=200)

    # ==============================================================================
    #     Target vs. Actual Area
    # ==============================================================================
    if args['uniformity_plot']:
        uniformity, samples = calc_uniformity(0, args, xi, yi, zi)
        fig = plt.figure(num="Uniformity Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Uniformity: ' + '{0:3.2%}'.format(uniformity))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Draw contour line
        cs = plt.contour(xi, yi, zi, [args['power_boundary_percentage'] * np.max(zi)],
                         alpha=1.0,
                         colors=colors.rgb2hex(cmap(args['power_boundary_percentage'])))
        manual_locations = [(0, np.max(yi))]
        plt.clabel(cs, [args['power_boundary_percentage']*np.max(zi)],
                   inline=True,
                   fmt='%1.2f',
                   fontsize=16,
                   manual=manual_locations)

        # Showing the Uniformity Points
        labels = np.around(samples[:, 2], decimals=3)
        sample_points = []
        for label, x, y, in zip(labels, samples[:, 0], samples[:, 1]):
            # Generate labels
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy=(x, y),
                xytext = (x, y),
                textcoords = 'offset points',
                ha = 'center',
                va = 'center',
                bbox = dict(boxstyle='round,pad=0.5',
                            facecolor='yellow',
                            alpha=0.7),
                arrowprops = dict(arrowstyle='->',
                                  connectionstyle='arc3, rad=0'))
            # Generate circle
            circle = Circle((x, y), args['aperture']/2)
            sample_points.append(circle)
        color = samples[:, 2]
        p = PatchCollection(sample_points, cmap=cmap, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)
        ax.add_patch(rectangle)

        # draw color bar
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(p, cax=cax)
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " Uniformity Plot", dpi=200)

    # =============================================================================
    #     #Surface Plot
    # =============================================================================
    if args['surface_plot']:
        fig = plt.figure(num="Surface Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Surface Plot')
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, linewidth=0, shade=True)
        ax.set_zlim(0, 1)

        x_surf_ticks = x_ticks[1::2]
        y_surf_ticks = y_ticks[1::2]

        ax.set_xticks(x_surf_ticks)
        ax.set_yticks(y_surf_ticks)
        ax.zaxis.set_major_locator(LinearLocator(11))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf)
        plt.axis('equal', adjustable='box')
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " surface plot", dpi=200)

    # ==============================================================================
    #     #Long Axis Plot
    # ==============================================================================
    if args['long_axis_plot']:
        if args['csv_export']:
            with open(args['scan_name']+" long axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(xi[0, :], long_axis_power)
                for row in rows:
                    writer.writerow(row)
            output.close()

        fig = plt.figure(num="Long Axis Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Long Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        ax.set_xticks(x_ticks)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        plt.axis([int(np.min(xi)), int(np.max(xi)), 0, np.max(long_axis_power)*1.1])
        ax.fill_between(xi[0, :], long_axis_power, 0, color=colors.rgb2hex(cmap(.5)), alpha=0.3)
        ax.plot(xi[0, :], long_axis_power, color=colors.rgb2hex(cmap(.5)))
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " long axis plot", dpi=200)

    # ==============================================================================
    #     #Short Axis Plot
    # ==============================================================================
    if args['short_axis_plot']:
        if args['csv_export']:
            with open(args['scan_name']+" short axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(yi[:, 0], args['short_axis_power'])
                for row in rows:
                    writer.writerow(row)
            output.close()
        fig = plt.figure(num="Short Axis Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Short Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        ax.set_xticks(y_ticks)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        #ax.zaxis.set_major_locator(LinearLocator(9))
        #ax.yaxis.set_major_locator(LinearLocator(9))
        plt.axis([int(np.min(yi)), int(np.max(yi)), 0, np.max(short_axis_power) * 1.1])
        ax.fill_between(yi[:, 0], short_axis_power, 0, color=colors.rgb2hex(cmap(0.5)), alpha=0.3)
        ax.plot(yi[:, 0], short_axis_power, color=colors.rgb2hex(cmap(.5)))
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " short axis plot", dpi=200)

    # ==============================================================================
    #     Generate Heat Map and Uniformity Plot Overlapped (Special Request)
    # ==============================================================================
    if args['heat_map_and_uniformity_plot']:
        fig = plt.figure(num="Heat Map and Uniformity Plot ("
                             + args['scan_name'] + ")")
        divider = make_axes_locatable(plt.gca())
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Heat Map And Uniformity Plot')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        plt.pcolormesh(xi, yi, zi, cmap=cmap)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])

        # draw rectangle
        ax.add_patch(rectangle)

        uniformity, samples = calc_uniformity(0, args, xi, yi, zi)
        labels = np.around(samples[:, 2], decimals=3)
        sample_points = []
        for label, x, y, in zip(labels, samples[:, 0], samples[:, 1]):
            # generate labels
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy=(x, y), xytext = (x, y),
                textcoords = 'offset points', ha = 'center', va = 'center',
                bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1),
                arrowprops =
                dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
            # generate circle
            circle = Circle((x, y), args['aperture']/2)
            sample_points.append(circle)
        color = samples[:, 2]

        # Changing the gray colormap based on reverse or non-reverse base cmap
        if sum(cmap(1))/len(cmap(1)) > 0.8:
            gray_cmap = plt.set_cmap('gist_gray')
        else:
            gray_cmap = plt.set_cmap('gist.yarg')
        p = PatchCollection(sample_points, cmap=gray_cmap, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)

        # drawing color-bar for heat map
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)

        # drawing color-bar for uniformity
        cax = divider.append_axes("left", size="5%", pad=0.5)
        cb = plt.colorbar(p, cax=cax)
        cb.ax.yaxis.set_ticks_position('left')
        if args['auto_save_figures']:
            savefig(args['scan_name'] + "heat map and uniformity plot", dpi=200)

    # ==============================================================================
    #     Generate Interpolated plot that goes corner to corner
    # ==============================================================================
    if args['diagonal_axis_plot']:
        try:
            theta = np.arctan([args['target_height']/args['target_width']])
        except ZeroDivisionError:
            theta = np.arctan(np.inf)

        theta = -theta
        # convert 2D array into one long 1D array
        xi_f, yi_f, zi_f = xi.flatten(), yi.flatten(), zi.flatten()

        # Givens transformation matrix
        rot_matrix = np.matrix(np.squeeze([[np.cos(theta), np.sin(theta)],
                                           [-np.sin(theta), np.cos(theta)]]))

        # Generated a ? x 2 matrix of x and y coordinates for matrix operations
        coordinates = np.transpose(np.matrix(np.squeeze([[xi_f], [yi_f]])))

        # Performing the Givens Rotation
        rot_coordinates = coordinates * rot_matrix
        rot_coordinates = np.array(rot_coordinates)

        new_grid = np.zeros((len(rot_coordinates), 3))
        new_grid[:, :-1] = rot_coordinates
        new_grid[:, -1] = zi_f

        # Trimming down gird data to y-axis near y = 0
        new_grid = new_grid[abs(new_grid[:, 1]) < args['pixel_pitch']*100]

        # Trimming down grid-data to x-values within map profile
        new_grid = \
            new_grid[abs(new_grid[:, 0]) < (1.5 / 2) * np.sqrt(args['target_height']**2 + args['target_width']**2)]

        # Re-Interpolate the data
        xi_r, yi_r, zi_r = grid_interpolation(new_grid[:, 0], new_grid[:, 1], new_grid[:, 2], args['pixel_pitch'])

        diagonal_power_index = find_nearest_index(yi_r[:, 0], 0)
        diagonal_power = zi_r[diagonal_power_index, :] / np.max(zi_r[diagonal_power_index, :])

        x_axis_start = int(-0.75 * np.sqrt(args['target_width']**2 + args['target_height']**2))
        x_axis_end = abs(x_axis_start)

        x_ticks_endpoint = max(abs(x_axis_start), abs(x_axis_end))
        diagonal_ticks = np.arange(0, x_ticks_endpoint, args['grid_resolution'])
        diagonal_ticks = np.unique(np.array([-np.flipud(diagonal_ticks), diagonal_ticks]).flatten())

        fig = plt.figure(num="Diagonal Axis Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Diagonal Axis Scan')
        ax.set_ylabel('Normalized Output')
        ax.set_xlabel('Distance from Lamp Center (mm)')

        ax.set_xticks(diagonal_ticks)
        ax.set_yticks(np.arange(0, 1.1, .1))
        plt.axis([int(np.min(diagonal_ticks)), int(np.max(diagonal_ticks)), 0, np.max(diagonal_power)*1.1])
        ax.fill_between(xi_r[0, :], diagonal_power, 0, color=colors.rgb2hex(cmap(0.5)), alpha=0.3)
        ax.plot(xi_r[0, :], diagonal_power, color=colors.rgb2hex(cmap(.5)))

        if args['auto_save_figures']:
            savefig(args['scan_name'] + " diagonal axis plot", dpi=200)
    plt.show(True)


def main():
    args = parser.parse_arguments()
    process(args)


# ==============================================================================
#     Find the index in array that is closest to Value
# ==============================================================================
def find_nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def find_boundaries(position, values, boundary_value):
    boundary_index_1 = find_nearest_index(values, boundary_value)
    boundary1 = position[boundary_index_1]

    position = np.delete(position, boundary_index_1)
    values = np.delete(values, boundary_index_1)

    boundary_index_2 = find_nearest_index(values, boundary_value)
    while abs(boundary_index_2-boundary_index_1) < 5:
        position = np.delete(position, boundary_index_2)
        values = np.delete(values, boundary_index_2)
        boundary_index_2 = find_nearest_index(values, boundary_value)
    boundary2 = position[boundary_index_2]

    return boundary1, boundary2


def calc_uniformity(box_size_ratio, args, xi, yi, zi):
    target_width = args['target_width']
    target_height = args['target_height']
    aperture = args['aperture']
    x_s = (target_width*(1-box_size_ratio)-aperture)/2
    y_s = (target_height*(1-box_size_ratio)-aperture)/2

    sample_points_indexes = np.mat([
        [find_nearest_index(xi[0, :], -x_s), find_nearest_index(yi[:, 0], -y_s)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], -y_s)],
        [find_nearest_index(xi[0, :], x_s), find_nearest_index(yi[:, 0], -y_s)],
        [find_nearest_index(xi[0, :], -x_s), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], x_s), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], -x_s), find_nearest_index(yi[:, 0], y_s)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], y_s)],
        [find_nearest_index(xi[0, :], x_s), find_nearest_index(yi[:, 0], y_s)]])

    samples = np.zeros([0, 3])
    for n in range(len(sample_points_indexes)):
        new_row = np.array([xi[0, sample_points_indexes[n, 0]],
                            yi[sample_points_indexes[n, 1], 0],
                            zi.data[sample_points_indexes[n, 1],
                                    sample_points_indexes[n, 0]]])
        samples = np.vstack([samples, new_row])
    uniformity = (np.max(samples[:, 2]) - np.min(samples[:, 2])) / np.max(samples[:, 2])
    return uniformity, samples


def parse_row_raw_data(filename):
    # read the CSV file
    my_data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # parse the CSV file
    x = my_data[:, 0]
    y = my_data[:, 1]
    z = my_data[:, 2]
    return x, y, z


def parse_sim_data(filename):

    # Parsing header information
    with open(filename) as my_file:
        head = list(islice(my_file, 12))

    for idx, string in enumerate(head):
        head[idx] = string.replace('\n', '')

    # Determining the Scan Name based on header information
    title = head[3].split(': ')[-1]

    peak_irr = head[10].split(': ')[-1]
    irr_multiplier = 10 ** int(peak_irr.split('E+')[1].split(' ')[0])
    peak_irr = float(peak_irr.split('E+')[0]) * irr_multiplier

    total_power = head[11].split(': ')[-1]
    pow_multiplier = 10 ** int(total_power.split('E+')[1].split(' ')[0])
    total_power = float(total_power.split('E+')[0]) * pow_multiplier

    scan_name = title + ' ' + str(peak_irr)[:5] + 'W/cm' + chr(0x00B2) + ' ' + str(total_power)[:5] + 'W'

    # Determining the pixel pitch
    size = head[8].split(', ')[0].split(' ')
    pixels = head[8].split(', ')[1].split(' ')

    horizontal_pitch = float(int(float(size[1]))/int(float(pixels[1])))
    vertical_pitch = float(int(float(size[4]))/int(float(pixels[4])))

    # Implement some kind of failure mechanism in case of non-uniform pixel spacing
    if abs(horizontal_pitch - vertical_pitch) > 1:
        return
    else:
        pixel_pitch = horizontal_pitch

    sim_data = np.genfromtxt(filename, delimiter='\t', skip_header=23)

    xi = sim_data[0, 1:]*pixel_pitch
    yi = sim_data[1:, 0]*pixel_pitch

    sim_data = np.delete(sim_data, 0, 0)
    sim_data = np.delete(sim_data, 0, 1)

    xi, yi = np.meshgrid(xi, yi)
    zi = np.ma.array(sim_data / np.max(sim_data))

    if abs(horizontal_pitch - vertical_pitch) > 1:
        return xi, yi, zi, scan_name, pixel_pitch, 'pixel_pitch_fault'
    else:
        return xi, yi, zi, scan_name, pixel_pitch


def grid_interpolation(x, y, z, pixel_pitch):
    xi = np.arange(np.min(x), np.max(x)+pixel_pitch, pixel_pitch)
    yi = np.arange(np.min(y), np.max(y)+pixel_pitch, pixel_pitch)
    # Generating a regular grid to interpolate the data
    xi, yi = np.meshgrid(xi, yi)

    zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
    # handle NaN issues
    zi = np.nan_to_num(zi)
    # Scaling Z values should normalPlot be set to True
    zi = zi / np.max(zi)
    return xi, yi, zi


# Center Finding Portion ###
def center_origin(xi, yi, zi):
    # generating new variables to facilitate edge-finding
    x, y, z = sorted(np.unique(xi)), sorted(np.unique(yi)), zi.data

    # finding Y value that corresponds to highest power readings for all X
    highest_power = 0
    long_axis_power = z[0, :]
    for n in list(range(len(y))):
        total_power = sum(z[n, :])
        if total_power > highest_power:
            highest_power = total_power
            long_axis_power = z[n, :]

    # setting the value that will be used to determine the cutoff...
    x_boundary_value = np.min(long_axis_power) + (np.max(long_axis_power) - np.min(long_axis_power)) / 4

    # finding x Boundaries
    x_boundary_1, x_boundary_2 = find_boundaries(x, long_axis_power, x_boundary_value)

    # determining middle of lamp along X axis
    mid_point_x = np.average([x_boundary_1, x_boundary_2])
    xi = xi - mid_point_x

    short_axis_power = z[:, find_nearest_index(xi[0], 0)]
    y_boundary_value = np.min(short_axis_power) + (np.max(short_axis_power) - np.min(short_axis_power)) / 4

    y_boundary_1, y_boundary_2 = find_boundaries(y, short_axis_power, y_boundary_value)

    mid_point_y = np.average([y_boundary_1, y_boundary_2])
    yi = yi - mid_point_y
    # End Center Finding #
    return xi, yi, mid_point_x, mid_point_y, long_axis_power, short_axis_power

if __name__ == "__main__":
    main()
