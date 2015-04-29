# TODO Trim Domain such that there isn't a large
# TODO Generate timestamp for both encoder reading and power reading
# Using the known deltaT and the known velocity, recompute a more precise
# position that the sensor reading was taken at.

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import argparse

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
style.use('ggplot')
# insert after each print statement
# import sy_s
# sy_s.stdout.flush()


# Importing separate paring file
def parse_arguments():
    # default values should nothing be passed/specified
    default_file_name = "FJ800 30x63 1-SLM 10mm.csv"
    # default_file_name = "sampledata.csv"
    # used in the naming of the plots
    default_scan_name = 'Sample Data'

    # Doug's Simulated data?
    default_simulated_data = False

    # Dual Axis Plots
    default_heat_map = False
    default_contour_plot = True
    default_surface_plot = False
    default_uniformity_plot = True
    default_heat_map_and_uniformity_plot = False
    default_uniformity_vs_box_size_ratio_plot = False

    # Single Axis Plots
    default_long_axis_plot = False
    default_short_axis_plot = False
    default_diagonal_axis_plot = True

    # determines grid line spacing on contour plot
    default_grid_resolution = 10
    # number of contour plot lines to display
    default_contour_lines = 10

    # lamp boundary to render
    default_power_boundary = .90

    # lamp profile to evaluate
    # FJ800 - 100, 100
    # FJ100-75 - 75, 25
    default_target_width, default_target_height = 200, 100

    # default to 1mm, can be lowered for higher resolution plots
    default_pixel_pitch = .2

    # used in the uniformity calculation
    # if no aperture is used on sensor, keep at 12.5
    # 1mm & 5mm apertures are available and should be used only when requested
    default_aperture = 12.5

    # export CSV file contents
    default_csv_export = False

    # automatically save figures
    default_auto_save_figure = False

    # set default colormap
    # default_colormap = cm.cubehelix_r
    # default_colormap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
    default_colormap = 'red'
    default_reverse = False
    # for more info on argparse see: http://pymotw.com/2/argparse/
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', action='store', dest='filename',
                        default=default_file_name,
                        help='The file name of the csv file used for post processing')
    parser.add_argument('-n', '--name', action='store', dest='scan_name',
                        default=default_scan_name,
                        help='Naming the scan, the name will be applied to plot titles')
    parser.add_argument('-g', '--grid', action='store', dest='grid_resolution',
                        default=default_grid_resolution,
                        help='the spacing between grid lines',
                        type=int)
    parser.add_argument('-l', '--lines', action='store',
                        dest='contour_resolution',
                        default=default_contour_lines,
                        help='number of contour lines to draw on the contour plot',
                        type=int)
    parser.add_argument('-b', '--boundary', action='store',
                        dest='power_boundary_percentage',
                        default=default_power_boundary,
                        help='Percentage of max power to draw region relative to lamp profile',
                        type=float)
    parser.add_argument('-p', '--pitch', action='store',
                        dest='pixel_pitch',
                        default=default_pixel_pitch,
                        help='Distance between interpolated points',
                        type=float)
    parser.add_argument('-W', '--width', action='store',
                        dest='target_width',
                        default=default_target_width,
                        help='Desired Light Profile Width (mm)',
                        type=int)
    parser.add_argument('-H', '--height', action='store',
                        dest='target_height',
                        default=default_target_height,
                        help='Desired Light Profile Height (mm)',
                        type=int)
    parser.add_argument('-a', '--aperture', action='store',
                        dest='aperture',
                        default=default_aperture,
                        help='Diameter of aperture used on sensor',
                        type=float)
    parser.add_argument('-M', '--colormap', action='store',
                        dest='colormap',
                        default=default_colormap,
                        help='Color Map Scheme to use for plotting',
                        type=str)
    # boolean conditions
    parser.add_argument('-R', '--reverse', action='store_true',
                        dest='colormap_reverse',
                        default=default_reverse,
                        help='Reverse Color Map Scheme')
    parser.add_argument('-q', '--simulated', action='store_true',
                        dest='simulated_data',
                        default=default_simulated_data,
                        help="Pass argument if using data from Doug's Simulations")
    parser.add_argument('-c', '--contour', action='store_true',
                        dest='contour_plot',
                        default=default_contour_plot,
                        help='Generate Topograhical Plot')
    parser.add_argument('-m', '--heatmap', action='store_true',
                        dest='heat_map',
                        default=default_heat_map,
                        help='Generate Heat Map')
    parser.add_argument('-L', "--long", action='store_true',
                        dest='long_axis_plot',
                        default=default_long_axis_plot,
                        help='Generate Long Axis Plot')
    parser.add_argument('-S', "--short", action='store_true',
                        dest='short_axis_plot',
                        default=default_short_axis_plot,
                        help='Generate Short Axis Plot')
    parser.add_argument('-D', "--diagonal", action='store_true',
                        dest='diagAxisPlot',
                        default=default_diagonal_axis_plot,
                        help='Generate Diagonal Plot')
    parser.add_argument('-s', '--surface', action='store_true',
                        dest='surface_plot',
                        default=default_surface_plot,
                        help='Geenrate Surface Plot')
    parser.add_argument('-u', '--uniformity', action='store_true',
                        dest='uniformity_plot',
                        default=default_uniformity_plot,
                        help='Generate Uniformity Plot')
    parser.add_argument('-U', '--uniformityCheck', action='store_true',
                        dest='uniformity_vs_box_size_ratio_plot',
                        default=default_uniformity_vs_box_size_ratio_plot,
                        help='Plot to ensure the sample points used for uniformity calculation are accurate')
    parser.add_argument('-A', '--autoSave', action='store_true',
                        dest='autoSave',
                        default=default_auto_save_figure,
                        help='Auto Save Figures')
    parser.add_argument('-e', '--exportCSV', action='store_true',
                        dest='csv_export',
                        default=default_csv_export,
                        help='Export CSV Data from interpolated long and short axis scans')
    parser.add_argument('-z, --heatMapAndUniformity', action='store_true',
                        dest='heat_map_and_uniformity_plot',
                        default=default_heat_map_and_uniformity_plot,
                        help='Heat Map and Uniformity Plot super-positioned, a plot request from Garth')

    args = parser.parse_args()

    argument = {'filename': args.filename,
                'grid_resolution': args.grid_resolution,
                'contour_resolution': args.contour_resolution,
                'scan_name': args.scan_name,
                'power_boundary_percentage': args.power_boundary_percentage,
                'pixel_pitch': args.pixel_pitch,
                'target_width': args.target_width,
                'target_height': args.target_height,
                'contour_plot': args.contour_plot,
                'heat_map': args.heat_map,
                'long_axis_plot': args.long_axis_plot,
                'short_axis_plot': args.short_axis_plot,
                'diagonal_axis_plot': args.diagAxisPlot,
                'surface_plot': args.surface_plot,
                'uniformity_plot': args.uniformity_plot,
                'uniformity_vs_box_size_ratio_plot': args.uniformity_vs_box_size_ratio_plot,
                'aperture': args.aperture,
                'auto_save_figures': args.autoSave,
                'csv_export': args.csv_export,
                'heat_map_and_uniformity_plot': args.heat_map_and_uniformity_plot,
                'colormap': args.colormap,
                'colormap_reverse': args.colormap_reverse,
                'simulated_data': args.simulated_data}
    return argument


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

    color_maps = {'red': sns.dark_palette("red", reverse=args['colormap_reverse'], as_cmap=True),
                  'green': sns.dark_palette("green", reverse=args['colormap_reverse'], as_cmap=True),
                  'blue': sns.dark_palette("blue", reverse=args['colormap_reverse'], as_cmap=True),
                  'purple': sns.dark_palette("purple", reverse=args['colormap_reverse'], as_cmap=True),
                  'cube helix': cm.cubehelix,
                  'cube helix purple': sns.cubehelix_palette(light=1,
                                                             reverse=args['colormap_reverse'],
                                                             as_cmap=True)}

    # Setting The ColorMap
    cmap = color_maps[args['colormap']]

    # cmap.set_bad('r', 1.0)
    # ==============================================================================
    #     #determining grid-line spacing
    # ==============================================================================
    x_tick_boundary = args['grid_resolution'] * int(max(abs(np.min(xi)),
                                                        np.max(xi)) / args['grid_resolution'])
    x_ticks = np.arange(-x_tick_boundary,
                        x_tick_boundary + args['grid_resolution'],
                        args['grid_resolution'])
    y_tick_boundary = args['grid_resolution'] * int(max(abs(np.min(yi)),
                                                        np.max(yi))/args['grid_resolution'])
    y_ticks = np.arange(-y_tick_boundary,
                        y_tick_boundary + args['grid_resolution'],
                        args['grid_resolution'])

    # draw rectangle
    rectangle = Rectangle((-args['target_width']/2,
                           -args['target_height']/2),
                          args['target_width'],
                          args['target_height'],
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
            uniformity.append(calc_uniformity(i, args, xi, yi, zi)[0])
        fig = plt.figure(num="Uniformity vs. Box Ratio (" + args['scan_name']
                             + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Uniformity vs. Box Ratio Plot')
        ax.set_xlabel('Box Size Ratio')
        ax.set_ylabel('Uniformity (%)')
        ax.plot(box_size_ratio, uniformity)
        plt.axis([np.min(box_size_ratio), np.max(box_size_ratio),
                  0, np.max(uniformity)*1.1])
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " uniformity vs box ratio plot", dpi=200)

            # ==============================================================================
            #     #Plot Basic Heat Map
            # ==============================================================================
    if args['heat_map']:
        fig = plt.figure(num="Heat Map (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Heat Map')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pcolormesh(xi, yi, zi, cmap=cmap, shading='gouraud', alpha=0.8)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)
        ax.grid(True, which='both')
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " Heat Map", dpi=200)

            # ==============================================================================
            #     #Contour View
            # ==============================================================================
    if args['contour_plot']:
        fig = plt.figure(num="Contour Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Contour Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(x_ticks)
        plt.gca().set_yticks(y_ticks)
        plt.contour(xi, yi, zi, args['contour_resolution'],
                    linewidths=1, cmap=cmap)
        ax.add_patch(rectangle)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)
        if args['auto_save_figures']:
            savefig(args['scan_name'] + " contour plot", dpi=200)

            # ==============================================================================
            #     Target vs. Actual Area
            # ==============================================================================
    if args['uniformity_plot']:
        uniformity, samples = calc_uniformity(0, args, xi, yi, zi)

        fig = plt.figure(num="Uniformity Plot (" + args['scan_name'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scan_name'] + ' Uniformity: ' +
                     '{0:3.2%}'.format(uniformity))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(x_ticks)
        plt.gca().set_yticks(y_ticks)

        # Draw contour line
        cs = plt.contour(xi, yi, zi, [args['power_boundary_percentage']
                                      * np.max(zi)],
                         alpha=0.9,
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

        # draw colorbar
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
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, linewidth=0)

        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
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
        plt.gca().set_xticks(x_ticks)
        plt.gca().set_yticks(np.arange(0., 1.+1./args['grid_resolution'],
                                       1./args['grid_resolution']))
        plt.axis([int(np.min(xi)), int(np.max(xi)), 0, np.max(long_axis_power)*1.1])
        ax.fill(xi[0, :], long_axis_power,
                color=colors.rgb2hex(cmap(.5)),
                alpha=0.3)
        ax.plot(xi[0, :], long_axis_power,
                color=colors.rgb2hex(cmap(.5)))
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
        plt.gca().set_xticks(y_ticks)
        plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
        plt.axis([int(np.min(yi)), int(np.max(yi)), 0,
                  np.max(short_axis_power) * 1.1])
        ax.fill_between(yi[:, 0], short_axis_power, 0, color=colors.rgb2hex(cmap(0.5)),
                        alpha=0.3)
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
        ax.set_title(args['scan_name'] + ' Heat Map And Unifromity Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(x_ticks)
        plt.gca().set_yticks(y_ticks)
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
            gray_cmap = cm.Grey_s_r
        else:
            gray_cmap = cm.Grey_s
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
            #     Generate Interpoalted plot that goes corner to corner
            #
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
        new_grid = new_grid[abs(new_grid[:, 1]) < 30]

        # Trimming down grid-data to x-values within map profile
        new_grid = new_grid[abs(new_grid[:, 0]) < (1.1 / 2) * np.sqrt(args['target_height']**2 + args['target_width']**2)]

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

        plt.gca().set_xticks(diagonal_ticks)
        plt.axis([int(x_axis_start*1.5), int(x_axis_end*1.5),
                  0, np.max(diagonal_power)*1.1])
        ax.fill(xi_r[0, :], diagonal_power,
                color=colors.rgb2hex(cmap(0.5)), alpha=0.3)
        ax.plot(xi_r[0, :], diagonal_power, color=colors.rgb2hex(cmap(.5)))

        if args['auto_save_figures']:
            savefig(args['scan_name'] + " diagonal axis plot", dpi=200)
    plt.show(True)


def main():
    args = parse_arguments()
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
    with open(filename) as myfile:
        head = list(islice(myfile, 12))

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
    # interpolating using delaunay triangularization/natural neighbor

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
    boundary_value = max(long_axis_power)*.7

    # finding x Boundaries
    x_boundary_1, x_boundary_2 = find_boundaries(x, long_axis_power, boundary_value)

    # determining middle of lamp along X axis
    mid_point_x = np.average([x_boundary_1, x_boundary_2])
    xi = xi - mid_point_x

    short_axis_power = z[:, find_nearest_index(xi[0], 0)]
    y_boundary_1, y_boundary_2 = find_boundaries(y, short_axis_power, boundary_value)

    mid_point_y = np.average([y_boundary_1, y_boundary_2])
    yi = yi - mid_point_y
    # End Center Finding #
    return xi, yi, mid_point_x, mid_point_y, long_axis_power, short_axis_power

if __name__ == "__main__":
    main()
