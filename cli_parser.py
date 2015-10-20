__author__ = 'omoore'
# Importing separate paring file
import argparse
def parse_arguments():
    # default values should nothing be passesd/specified
    default_file_name = "FJ800 395nm 2mm.csv"
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
    default_target_width, default_target_height = 100, 100

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
    default_colormap = 'cube-helix'
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
                        help='Generate Topographical Plot')
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
                        dest='diagonal_axis_plot',
                        default=default_diagonal_axis_plot,
                        help='Generate Diagonal Plot')
    parser.add_argument('-s', '--surface', action='store_true',
                        dest='surface_plot',
                        default=default_surface_plot,
                        help='Generate Surface Plot')
    parser.add_argument('-u', '--uniformity', action='store_true',
                        dest='uniformity_plot',
                        default=default_uniformity_plot,
                        help='Generate Uniformity Plot')
    parser.add_argument('-U', '--uniformityCheck', action='store_true',
                        dest='uniformity_vs_box_size_ratio_plot',
                        default=default_uniformity_vs_box_size_ratio_plot,
                        help='Plot to ensure the sample points used for uniformity calculation are accurate')
    parser.add_argument('-A', '--autoSave', action='store_true',
                        dest='auto_save',
                        default=default_auto_save_figure,
                        help='Auto Save Figures')
    parser.add_argument('-e', '--exportCSV', action='store_true',
                        dest='csv_export',
                        default=default_csv_export,
                        help='Export CSV Data from interpolated long and short axis scans')
    parser.add_argument('-z, --heat_map_and_uniformity_plot', action='store_true',
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
                'diagonal_axis_plot': args.diagonal_axis_plot,
                'surface_plot': args.surface_plot,
                'uniformity_plot': args.uniformity_plot,
                'uniformity_vs_box_size_ratio_plot': args.uniformity_vs_box_size_ratio_plot,
                'aperture': args.aperture,
                'auto_save_figures': args.auto_save,
                'csv_export': args.csv_export,
                'heat_map_and_uniformity_plot': args.heat_map_and_uniformity_plot,
                'colormap': args.colormap,
                'colormap_reverse': args.colormap_reverse,
                'simulated_data': args.simulated_data}
    return argument
