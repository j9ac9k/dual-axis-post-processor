#TODO Trim Domain such that there isn't a large

#TODO Generate timestamp for both encoder reading and power reading
#Using the known deltaT and the known velocity, recompute a more precise
#position that the sensor reading was taken at.

#import sys
import csv
import re

#insert after each print statement
#sys.stdout.flush()

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
style.use('ggplot')

from matplotlib import cm
from matplotlib import colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import savefig

#Importing separate paring file
def parseArguments():
    #default values should nothing be passed/specified
    defaultFileName = "FJ800 30x63 1-SLM 10mm.csv"
    #defaultFileName = "sampledata.csv"
    #used in the naming of the plots
    defaultScanName = 'Sample Data'
    
    # Doug's Simulated data?
    defaultSimulatedData = False

    #Dual Axis Plots
    defaultHeatMap = False
    defaultContourPlot = True
    defaultSurfacePlot = False
    defaultUniformityPlot = True
    defaultHeatMapAndUniformityPlot = False
    defaultUniformityVsBoxSizeRatioPlot = False

    #Single Axis Plots
    defaultLongAxisPlot = False
    defaultShortAxisPlot = False
    defaultDiagonalAxisPlot = True

    #determines grid line spacing on contour plot
    defaultGridResolution = 10
    #number of contour plot lines to display
    defaultContourResolution = 10

    #lamp boundary to render
    defaultPowerBoundaryPercentage = .90

    #lamp profile to evaluate
    #FJ800 - 100, 100
    #FJ100-75 - 75, 25
    defaultTargetWidth, defaultTargetHeight = 200, 100

    #default to 1mm, can be lowered for higher resolution plots
    defaultPixelPitch = .2

    #used in the uniformity calculation
    #if no aperture is used on sensor, keep at 12.5
    #1mm & 5mm apertures are available and should be used only when requested
    defaultAperture = 12.5

    #export CSV file contents
    defaultCSVExport = False

    #automatically save figures
    defaultAutoSaveFigures = False

    #set default colormap
    #defaultColorMap = cm.cubehelix_r
    #defaultColorMap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
    defaultColorMap = 'red'
    defaultReverse = False
    #for more info on argparse see: http://pymotw.com/2/argparse/
    parser = argparse.ArgumentParser()


    parser.add_argument('-f', '--file', action='store', dest='fileName',
                        default=defaultFileName,
                        help='The file name of the csv file used for post processing')
    parser.add_argument('-n', '--name', action='store', dest='scanName',
                        default=defaultScanName,
                        help='Naming the scan, the name will be applied to plot titles')
    parser.add_argument('-g', '--grid', action='store', dest='gridResolution',
                        default=defaultGridResolution,
                        help='the spacing between grid lines',
                        type=int)

    parser.add_argument('-l', '--lines', action='store',
                        dest='contourResolution',
                        default=defaultContourResolution,
                        help='number of contour lines to draw on the contour plot',
                        type=int)

    parser.add_argument('-b', '--boundary', action='store',
                        dest='powerBoundaryPercentage',
                        default=defaultPowerBoundaryPercentage,
                        help='Percentage of max power to draw region relative to lamp profile',
                        type=float)
    parser.add_argument('-p', '--pitch', action='store',
                        dest='pixelPitch',
                        default=defaultPixelPitch,
                        help='Distance between interpolated points',
                        type=float)
    parser.add_argument('-W', '--width', action='store',
                        dest='targetWidth',
                        default=defaultTargetWidth,
                        help='Desired Light Profile Width (mm)',
                        type=int)
    parser.add_argument('-H', '--height', action='store',
                        dest='targetHeight',
                        default=defaultTargetHeight,
                        help='Desired Light Profile Height (mm)',
                        type=int)
    parser.add_argument('-a', '--aperture', action='store',
                        dest='aperture',
                        default=defaultAperture,
                        help='Diameter of aperture used on sensor',
                        type=float)
    parser.add_argument('-M', '--colormap', action='store',
                        dest='colorMap',
                        default=defaultColorMap,
                        help='Color Map Scheme to use for plotting',
                        type=str)

    #boolean conditions
    parser.add_argument('-R', '--reverse', action='store_true',
                        dest='colorMapReverse',
                        default=defaultReverse,
                        help='Reverse Color Map Scheme')
    parser.add_argument('-q', '--simulated', action='store_true',
                        dest='simulatedData',
                        default=defaultSimulatedData,
                        help="Pass argument if using data from Doug's Simulations")
    parser.add_argument('-c', '--contour', action='store_true',
                        dest='contourPlot',
                        default=defaultContourPlot,
                        help='Generate Topograhical Plot')
    parser.add_argument('-m', '--heatmap', action='store_true',
                        dest='heatMap',
                        default=defaultHeatMap,
                        help='Generate Heat Map')
    parser.add_argument('-L', "--long", action='store_true',
                        dest='longAxisPlot',
                        default=defaultLongAxisPlot,
                        help='Generate Long Axis Plot')
    parser.add_argument('-S', "--short", action='store_true',
                        dest='shortAxisPlot',
                        default=defaultShortAxisPlot,
                        help='Generate Short Axis Plot')
    parser.add_argument('-D', "--diagonal", action='store_true',
                        dest='diagAxisPlot',
                        default=defaultDiagonalAxisPlot,
                        help='Generate Diagonal Plot')
    parser.add_argument('-s', '--surface', action='store_true',
                        dest='surfacePlot',
                        default=defaultSurfacePlot,
                        help='Geenrate Surface Plot')
    parser.add_argument('-u', '--uniformity', action='store_true',
                        dest='uniformityPlot',
                        default=defaultUniformityPlot,
                        help='Generate Uniformity Plot')
    parser.add_argument('-U', '--uniformityCheck', action='store_true',
                        dest='uniformityVsBoxSizeRatioPlot',
                        default=defaultUniformityVsBoxSizeRatioPlot,
                        help='Plot to ensure the sample points used for uniformity calculation are accurate')
    parser.add_argument('-A', '--autoSave', action='store_true',
                        dest='autoSave',
                        default=defaultAutoSaveFigures,
                        help='Auto Save Figures')
    parser.add_argument('-e', '--exportCSV', action='store_true',
                        dest='csvExport',
                        default=defaultCSVExport,
                        help='Export CSV Data from interpolated long and short axis scans')
    parser.add_argument('-z, --heatMapAndUniformity', action='store_true',
                        dest='heatMapAndUniformityPlot',
                        default=defaultHeatMapAndUniformityPlot,
                        help='Heat Map and Uniformity Plot super-positioned, a plot request from Garth')

    args = parser.parse_args()
    argument = {}
    argument['fileName'] = args.fileName
    argument['gridResolution'] = args.gridResolution
    argument['contourResolution'] = args.contourResolution
    argument['scanName'] = args.scanName
    argument['powerBoundaryPercentage'] = args.powerBoundaryPercentage
    argument['pixelPitch'] = args.pixelPitch
    argument['targetWidth'] = args.targetWidth
    argument['targetHeight'] = args.targetHeight
    argument['contourPlot'] = args.contourPlot
    argument['heatMap'] = args.heatMap
    argument['longAxisPlot'] = args.longAxisPlot
    argument['shortAxisPlot'] = args.shortAxisPlot
    argument['diagonalAxisPlot'] = args.diagAxisPlot
    argument['surfacePlot'] = args.surfacePlot
    argument['uniformityPlot'] = args.uniformityPlot
    argument['uniformityVsBoxSizeRatioPlot'] = args.uniformityVsBoxSizeRatioPlot
    argument['aperture'] = args.aperture
    argument['autoSaveFigures'] = args.autoSave
    argument['csvExport'] = args.csvExport
    argument['heatMapAndUniformityPlot'] = args.heatMapAndUniformityPlot
    argument['colorMap'] = args.colorMap
    argument['colorMapReverse'] = args.colorMapReverse
    argument['simulatedData'] = args.simulatedData
    return argument


def process(args):
    #close all currently open plots
    plt.close("all")
    if args['simulatedData']:
        xi, yi, zi, args['scanName'], args['pixelPitch'], fault = parseSimData(args.get('fileName', None))
        if fault == 'pixel_pitch_fault':
            return fault

    else:
        x, y, z = parseRowRawData(args.get('fileName', None))


        xi, yi, zi = gridInterpolation(x, y, z, args.get('pixelPitch', None))
    xi, yi, xOffset, yOffset, longAxisPower, shortAxisPower = centerOrigin(xi, yi, zi)

    color_maps = {'red': sns.dark_palette("red", reverse=args.get('colorMapReverse', None), as_cmap=True),
                  'green': sns.dark_palette("green", reverse=args.get('colorMapReverse', None), as_cmap=True),
                  'blue': sns.dark_palette("blue", reverse=args.get('colorMapReverse', None), as_cmap=True),
                  'purple': sns.dark_palette("purple", reverse=args.get('colorMapReverse', None), as_cmap=True),
                  'cube helix': cm.cubehelix,
                  'cube helix purple': sns.cubehelix_palette(light=1, reverse=args.get('colorMapReverse', None), as_cmap=True)}


    #Setting The ColorMap
    cmap = color_maps[args['colorMap']]

    #cmap.set_bad('r', 1.0)
#==============================================================================
#     #determining grid-line spacing
#==============================================================================
    xTickBoundary = args['gridResolution'] * int(max(abs(np.min(xi)),
                                                     np.max(xi)) / args['gridResolution'])
    xTicks = np.arange(-xTickBoundary,
                       xTickBoundary + args['gridResolution'],
                       args['gridResolution'])
    yTickBoundary = args['gridResolution'] * int(max(abs(np.min(yi)),
                                                     np.max(yi))/args['gridResolution'])
    yTicks = np.arange(-yTickBoundary,
                       yTickBoundary + args['gridResolution'],
                       args['gridResolution'])

    #draw rectangle
    rectangle = Rectangle((-args['targetWidth']/2,
                            -args['targetHeight']/2),
                           args['targetWidth'],
                           args['targetHeight'],
                           #alpha=0.1,
                           linewidth=3,
                           fill=False,
                           edgecolor='black',
                           ls='dashed')
#==============================================================================
#     #calculate uniformity as a function of box size ratio
#==============================================================================
    if args['uniformityVsBoxSizeRatioPlot']:
        boxSizeRatio = np.arange(-0.10, .330, 1 / (args['pixelPitch'] * max(args['targetWidth'],
                                                                            args['targetHeight'])))
        uniformity = []
        for i in boxSizeRatio:
            uniformity.append(calcUniformity(i, args, xi, yi, zi)[0])
        fig = plt.figure(num="Uniformity vs. Box Ratio (" + args['scanName']
                         + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Uniformity vs. Box Ratio Plot')
        ax.set_xlabel('Box Size Ratio')
        ax.set_ylabel('Uniformity (%)')
        ax.plot(boxSizeRatio, uniformity)
        plt.axis([np.min(boxSizeRatio), np.max(boxSizeRatio),
                 0, np.max(uniformity)*1.1])
        #plt.grid()
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " uniformity vs box ratio plot", dpi=200)

#==============================================================================
#     #Plot Basic Heat Map
#==============================================================================
    if args['heatMap']:
        fig = plt.figure(num="Heat Map (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Heat Map')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pcolormesh(xi, yi, zi, cmap=cmap, shading='gouraud', alpha=0.8)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)
        ax.grid(True, which='both')
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " Heat Map", dpi=200)

#==============================================================================
#     #Contour View
#==============================================================================
    if args['contourPlot']:
        fig = plt.figure(num="Contour Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Contour Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)
        CS = plt.contour(xi, yi, zi, args['contourResolution'],
                         linewidths=1, cmap=cmap)
        ax.add_patch(rectangle)
        #plt.grid()
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " contour plot", dpi=200)

#==============================================================================
#     Target vs. Actual Area
#==============================================================================
    if args['uniformityPlot']:
        uniformity, samples = calcUniformity(0, args, xi, yi, zi)
        
        fig = plt.figure(num="Uniformity Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Uniformity: ' +
                     '{0:3.2%}'.format(uniformity))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)
        
        # Draw contour line
        CS = plt.contour(xi, yi, zi, [args['powerBoundaryPercentage']
                         * np.max(zi)],
                         alpha=0.9,
                         colors=colors.rgb2hex(cmap(args['powerBoundaryPercentage'])))
        manual_locations = [(0, np.max(yi))]
        plt.clabel(CS, [args['powerBoundaryPercentage']*np.max(zi)],
                   inline=True,
                   fmt='%1.2f',
                   fontsize=16,
                   manual=manual_locations)

        # Showing the Uniformity Points
        labels = np.around(samples[:, 2], decimals=3)
        samplePoints = []
        for label, x, y, in zip(labels, samples[:, 0], samples[:, 1]):
            # Generate labels
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy=(x, y), xytext = (x, y),
                textcoords = 'offset points', ha = 'center', va = 'center',
                bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops = dict(arrowstyle='->',
                                  connectionstyle='arc3, rad=0'))
            # Generate circle
            circle = Circle((x, y), args['aperture']/2)
            samplePoints.append(circle)
        color = samples[:, 2]
        p = PatchCollection(samplePoints, cmap=cmap, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)
        ax.add_patch(rectangle)

        #draw colorbar
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(p, cax=cax)
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " Uniformity Plot", dpi=200)

# =============================================================================
#     #Surface Plot
# =============================================================================
    if args['surfacePlot']:
        fig = plt.figure(num="Surface Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Surface Plot')
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, linewidth=0)

        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf)
        plt.axis('equal', adjustable='box')
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " surface plot", dpi=200)

#==============================================================================
#     #Long Axis Plot
#==============================================================================
    if args['longAxisPlot']:
        if args['csvExport']:
            with open(args['scanName']+" long axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(xi[0, :], longAxisPower)
                for row in rows:
                    writer.writerow(row)
            output.close()
        
        fig = plt.figure(num="Long Axis Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Long Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(np.arange(0., 1.+1./args['gridResolution'],
                             1./args['gridResolution']))
        plt.axis([int(np.min(xi)), int(np.max(xi)), 0, np.max(longAxisPower)*1.1])
        ax.fill(xi[0, :], longAxisPower,
                color=colors.rgb2hex(cmap(.5)),               
                alpha=0.3)
        ax.plot(xi[0, :], longAxisPower,
                color=colors.rgb2hex(cmap(.5)))
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " long axis plot", dpi=200)

#==============================================================================
#     #Short Axis Plot
#==============================================================================
    if args['shortAxisPlot']:
        if args['csvExport']:
            with open(args['scanName']+" short axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(yi[:, 0], args['shortAxisPower'])
                for row in rows:
                    writer.writerow(row)
            output.close()
        fig = plt.figure(num="Short Axis Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Short Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        plt.gca().set_xticks(yTicks)
        plt.axis([int(np.min(yi)), int(np.max(yi)), 0,
                 np.max(shortAxisPower) * 1.1])
        ax.fill_between(yi[:, 0], shortAxisPower, 0, color=colors.rgb2hex(cmap(0.5)),
                alpha=0.3)
        ax.plot(yi[:, 0], shortAxisPower, color=colors.rgb2hex(cmap(.5)))
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " short axis plot", dpi=200)

#==============================================================================
#     Generate Heat Map and Uniformity Plot Overlapped (Special Request)
#==============================================================================
    if args['heatMapAndUniformityPlot']:
        fig = plt.figure(num="Heat Map and Uniformity Plot ("
                         + args['scanName'] + ")")
        divider = make_axes_locatable(plt.gca())
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Heat Map And Unifromity Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)
        plt.pcolormesh(xi, yi, zi, cmap=cmap)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        
        # draw rectangle
        ax.add_patch(rectangle)

        uniformity, samples = calcUniformity(0, args, xi, yi, zi)
        labels = np.around(samples[:, 2], decimals=3)
        samplePoints = []
        for label, x, y, in zip(labels, samples[:, 0], samples[:, 1]):
            #generate labels
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy=(x, y), xytext = (x, y),
                textcoords = 'offset points', ha = 'center', va = 'center',
                bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1),
                arrowprops =
                dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
            #generate circle
            circle = Circle((x, y), args['aperture']/2)
            samplePoints.append(circle)
        color = samples[:, 2]
        
        # Changing the gray colormap based on reverse or non-reverse base cmap
        if sum(cmap(1))/len(cmap(1)) > 0.8:
            grayCmap = cm.Greys_r
        else:
            grayCmap = cm.Greys
        p = PatchCollection(samplePoints, cmap=grayCmap, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)

        #drawing color-bar for heat map
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax=cax)

        #drawing color-bar for uniformity
        cax = divider.append_axes("left", size="5%", pad=0.5)
        cb = plt.colorbar(p, cax=cax)
        cb.ax.yaxis.set_ticks_position('left')
        if args['autoSaveFigures']:
            savefig(args['scanName'] + "heat map and uniformity plot", dpi=200)

#==============================================================================
#     Generate Interpoalted plot that goes corner to corner
#
#==============================================================================
    if args['diagonalAxisPlot']:
        try:
            theta = np.arctan([args['targetHeight']/args['targetWidth']])
        except ZeroDivisionError:
            theta = np.arctan[(np.inf)]

        theta = -theta
        # convert 2D array into one long 1D array
        xiF, yiF, ziF = xi.flatten(), yi.flatten(), zi.flatten()
        
        #Givens transformation matrix
        rotMatrix = np.matrix(np.squeeze([[np.cos(theta), np.sin(theta)],
                                          [-np.sin(theta), np.cos(theta)]]))

        # Generated a ? x 2 matrix of x and y coordinates for matrix operations
        coordinates = np.transpose(np.matrix(np.squeeze([[xiF], [yiF]])))
        
        # Performing the Givens Rotation
        rotCoordinates = coordinates * rotMatrix
        rotCoordinates = np.array(rotCoordinates)
        
        newGrid = np.zeros((len(rotCoordinates),3))
        newGrid[:,:-1] = rotCoordinates
        newGrid[:,-1] = ziF
        newGrid = newGrid[abs(newGrid[:, 1]) < 15]

        # Re-Interpolate the data
        xiR, yiR, ziR = gridInterpolation(newGrid[:,0],
                                          newGrid[:,1], 
                                          newGrid[:,2], 
                                          args.get('pixelPitch', None))

        diagPowerIndex = find_nearest_index(yiR[:, 0], 0)
        diagPower = ziR[diagPowerIndex, :] / np.max(ziR[diagPowerIndex, :])

        xAxisStart = int(-0.75 * np.sqrt(args['targetWidth']**2 + args['targetHeight']**2))
        xAxisEnd = abs(xAxisStart)

        xTicksEndpoints = min(abs(xAxisStart), abs(xAxisEnd))

        diagTicks = np.arange(-xTicksEndpoints, 
                              xTicksEndpoints,
                              args['gridResolution'])

        print(diagTicks)
        fig = plt.figure(num="Diagonal Axis Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Diagonal Axis Scan')
        ax.set_ylabel('Normalized Output')
        ax.set_xlabel('Distance from Lamp Center (mm)')
        
        plt.gca().set_xticks(diagTicks)
        plt.axis([int(xAxisStart*1.5), int(xAxisEnd*1.5), 
                  0, np.max(diagPower)*1.1])      
        ax.fill(xiR[0, :], diagPower,
                color=colors.rgb2hex(cmap(0.5)), alpha=0.3)
        ax.plot(xiR[0, :], diagPower, color=colors.rgb2hex(cmap(.5)))

        if args['autoSaveFigures']:
            savefig(args['scanName'] + " diagonal axis plot", dpi=200)
    plt.show(True)

def main():
    args = parseArguments()
    process(args)

#==============================================================================
#     Find the index in array that is closest to Value
#==============================================================================
def find_nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def find_boundaries(position, values, boundaryValue):
    boundaryIndex1 = find_nearest_index(values, boundaryValue)
    boundary1 = position[boundaryIndex1]

    position = np.delete(position, boundaryIndex1)
    values = np.delete(values, boundaryIndex1)

    boundaryIndex2 = find_nearest_index(values, boundaryValue)
    while abs(boundaryIndex2-boundaryIndex1) < 5:
            position = np.delete(position, boundaryIndex2)
            values = np.delete(values, boundaryIndex2)
            boundaryIndex2 = find_nearest_index(values, boundaryValue)
    boundary2 = position[boundaryIndex2]

    return boundary1, boundary2


def calcUniformity(boxSizeRatio, args, xi, yi, zi):
    targetWidth = args['targetWidth']
    targetHeight = args['targetHeight']
    aperture = args['aperture']
    xS = (targetWidth*(1-boxSizeRatio)-aperture)/2
    yS = (targetHeight*(1-boxSizeRatio)-aperture)/2

    samplePointsIndexes = np.mat([
        [find_nearest_index(xi[0, :], -xS), find_nearest_index(yi[:, 0], -yS)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], -yS)],
        [find_nearest_index(xi[0, :], xS), find_nearest_index(yi[:, 0], -yS)],
        [find_nearest_index(xi[0, :], -xS), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], xS), find_nearest_index(yi[:, 0], 0)],
        [find_nearest_index(xi[0, :], -xS), find_nearest_index(yi[:, 0], yS)],
        [find_nearest_index(xi[0, :], 0), find_nearest_index(yi[:, 0], yS)],
        [find_nearest_index(xi[0, :], xS), find_nearest_index(yi[:, 0], yS)]])

    samples = np.empty([0, 3])
    for n in range(len(samplePointsIndexes)):
        newRow = np.array([xi[0, samplePointsIndexes[n, 0]],
                           yi[samplePointsIndexes[n, 1], 0],
                           zi.data[samplePointsIndexes[n, 1],
                                   samplePointsIndexes[n, 0]]])
        samples = np.vstack([samples, newRow])
    uniformity = (np.max(samples[:, 2]) - np.min(samples[:, 2])) / \
        np.max(samples[:, 2])
    return uniformity, samples


def parseRowRawData(fileName):
    #read the CSV file
    my_data = np.genfromtxt(fileName, delimiter=',', skip_header=1)

    #parse the CSV file
    x = my_data[:, 0]
    y = my_data[:, 1]
    z = my_data[:, 2]
    return x, y, z


def parseSimData(fileName):

    #Parsing header information
    with open(fileName) as myfile:
        head = list(islice(myfile, 12))

    for idx, string in enumerate(head):
        head[idx] = string.replace('\n', '')
    
    #Determining the Scan Name based on header information
    title = head[3].split(': ')[-1]
    

    peak_irr = head[10].split(': ')[-1]
    irr_multiplier = 10 ** int(peak_irr.split('E+')[1].split(' ')[0])
    peak_irr = float(peak_irr.split('E+')[0]) * irr_multiplier

    total_power = head[11].split(': ')[-1]
    pow_multiplier = 10 ** int(total_power.split('E+')[1].split(' ')[0])
    total_power = float(total_power.split('E+')[0]) * pow_multiplier

    scanName = title + ' ' + str(peak_irr)[:5] + 'W/cm' + chr(0x00B2) + ' ' + str(total_power)[:5] + 'W'


    #Determining the pixel pitch

    size = head[8].split(', ')[0].split(' ')
    pixels = head[8].split(', ')[1].split(' ')

    horizontal_pitch = float(int(float(size[1]))/int(float(pixels[1])))
    vertical_pitch = float(int(float(size[4]))/int(float(pixels[4])))

    # Implement some kind of failure mechanism in case of non-uniform pixel spacing
    if abs(horizontal_pitch - vertical_pitch) > 1:
        pass
    else:
        pixelPitch = horizontal_pitch

    sim_data = np.genfromtxt(fileName, delimiter='\t', skip_header=23)

    xi = sim_data[0, 1:]*pixelPitch
    yi = sim_data[1:, 0]*pixelPitch
    
    sim_data = np.delete(sim_data, 0, 0)
    sim_data = np.delete(sim_data, 0, 1)

    xi, yi = np.meshgrid(xi, yi)
    zi = np.ma.array(sim_data / np.max(sim_data))

    return xi, yi, zi, scanName, pixelPitch


def gridInterpolation(x, y, z, pixelPitch):
    xi = np.arange(np.min(x), np.max(x)+pixelPitch, pixelPitch)
    yi = np.arange(np.min(y), np.max(y)+pixelPitch, pixelPitch)
    #Generating a regular grid to interpolate the data
    xi, yi = np.meshgrid(xi, yi)
    #interpolating using delaunay triangularization/natural neighbor
    zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
    #handle NaN issues
    zi = np.nan_to_num(zi)

    #Scaling Z values should normalPlot be set to True
    zi = zi / np.max(zi)
    return xi, yi, zi



   ### Center Finding Portion ###
def centerOrigin(xi, yi, zi):
    #generating new variables to facilitate edge-finding
    X, Y, Z = sorted(np.unique(xi)), sorted(np.unique(yi)), zi.data

    #finding Y value that corresponds to highest power readings for all X
    highestPower = 0
    for n in list(range(len(Y))):
        totalPower = sum(Z[n, :])
        if totalPower > highestPower:
            highestPower = totalPower
            longAxisPower = Z[n, :]

    #setting the value that will be used to determine the cutoff...
    boundaryValue = max(longAxisPower)*.7

    #finding x Boundaries
    xBoundary1, xBoundary2 = find_boundaries(X, longAxisPower, boundaryValue)

    #determining middle of lamp along X axis
    midPointX = np.average([xBoundary1, xBoundary2])
    xi = xi - midPointX

    shortAxisPower = Z[:, find_nearest_index(xi[0], 0)]
    yBoundary1, yBoundary2 = find_boundaries(Y, shortAxisPower, boundaryValue)

    midPointY = np.average([yBoundary1, yBoundary2])
    yi = yi - midPointY
    ### End Center Finding ###
    return xi, yi, midPointX, midPointY, longAxisPower, shortAxisPower

if __name__ == "__main__":
    main()
