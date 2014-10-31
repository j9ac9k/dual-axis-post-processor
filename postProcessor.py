#TODO Trim Domain such that there isn't a large 

#TODO Generate timestamp for both encoder reading and power reading
#Using the known deltaT and the known velocity, recompute a more precise
#position that the sensor reading was taken at.

#import sys
import csv
#import argparse
#insert after each print statement
#sys.stdout.flush()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import savefig

import argparse
#Importing seperate paring file


#default values should nothing be passed/specified

    
def main():
    #close all currently open plots     
    plt.close("all")                
    args = parseArguments()

    x, y, z = parseRowRawData(args.get('fileName', None))
    xi, yi, zi = gridInterpolation(x, y, z, args.get('pixelPitch', None))
    xi, yi, xOffset, yOffset, longAxisPower, shortAxisPower = centerOrigin(xi, yi, zi, args['powerBoundaryPercentage'])
    
#==============================================================================
#     secondaryUniformityScan = True
#     if secondaryUniformityScan:
#         coordinates = [(calcUniformity(0)[1][:,0] + xOffset).tolist(), (calcUniformity(0)[1][:,1] + yOffset).tolist()]
#==============================================================================
    
    
    #Setting The ColorMap
    cmap = cm.jet_r
    #cmap = cm.spectral
    cmap.set_bad('r', 1.0)
    
#==============================================================================
#     #determining grid-line spacing
#==============================================================================
    xTickBoundary = args['gridResolution']*int(max(abs(np.min(xi)), np.max(xi))/args['gridResolution'])
    xTicks = np.arange(-xTickBoundary, xTickBoundary+args['gridResolution'], args['gridResolution'])
    
    yTickBoundary = args['gridResolution']*int(max(abs(np.min(yi)), np.max(yi))/args['gridResolution'])
    yTicks = np.arange(-yTickBoundary, yTickBoundary+args['gridResolution'], args['gridResolution'])
    
#==============================================================================
#     #calculate uniformity as a function of box size ratio
#==============================================================================
    if args['uniformityVsBoxSizeRatioPlot']:
        boxSizeRatio = np.arange(-0.10, .330, 1/(args['pixelPitch']*max(args['targetWidth'],args['targetHeight'])))
        uniformity = []
        for i in boxSizeRatio:
            uniformity.append(calcUniformity(i)[0])
        fig = plt.figure(num = "Uniformity vs. Box Ratio (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(args['scanName'] + ' Uniformity vs. Box Ratio Plot')    
        ax.set_xlabel('Box Size Ratio')
        ax.set_ylabel('Uniformity (%)')
        ax.plot(boxSizeRatio, uniformity)
        plt.axis([np.min(boxSizeRatio), np.max(boxSizeRatio), 0, np.max(uniformity)*1.1])  
        plt.grid()
        if args['autoSaveFigures']:
            savefig(args['scanName'] + " uniformity vs box ratio plot")
        
#==============================================================================
#     #Plot Basic Contour Plot
#==============================================================================
    if args['contourPlot']:
        fig = plt.figure(num = "Contour Plot (" + args['scanName'] + ")")
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Contour Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pcolormesh(xi, yi, zi, cmap = cmap)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax = cax)
        if autoSaveFigures:
            savefig(scanName + " contour plot")
    
#==============================================================================
#     #Topographical View
#==============================================================================
    if topographicalPlot:
        fig = plt.figure(num = "Topographical Plot (" + scanName + ")")
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Topographical Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)
        CS = plt.contour(xi, yi, zi, topographicalResolution, linewidths=1, cmap = cmap)
        ax.add_patch(Rectangle((-targetWidth/2, -targetHeight/2), targetWidth, targetHeight, fill=False, ls='dashed'))
        plt.grid()
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax = cax)
        if autoSaveFigures:
            savefig(scanName + " topographical plot")
        
#==============================================================================
#     Target vs. Actual Area
#==============================================================================
    if uniformityPlot:
        uniformity, samples = calcUniformity(0)
        fig = plt.figure(num = "Uniformity Plot (" + scanName + ")")    
        ax = fig.add_subplot(111) 
        ax.set_title(scanName + ' Uniformity: ' + '{0:3.2%}'.format(uniformity))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)        
        #draw contour line    
        CS = plt.contour(xi, yi, zi, [powerBoundaryPercentage*np.max(zi)], alpha = 0.5)
        manual_locations = [(0, np.max(yi))]    
        plt.clabel(CS, [powerBoundaryPercentage*np.max(zi)],
               inline=True,
               fmt='%1.2f',
               fontsize=16,
               manual = manual_locations)
    
        labels = np.around(samples[:,2], decimals=3)
        samplePoints = []
        for label, x, y, in zip(labels, samples[:,0], samples[:,1]):
            #generate labels        
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy = (x, y), xytext = (x, y),
                textcoords = 'offset points', ha = 'center', va = 'center',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))
            #generate circle
            circle = Circle((x, y), aperture/2)
            samplePoints.append(circle)
        color = samples[:,2]
        p = PatchCollection(samplePoints, cmap = cm.jet_r, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)
     
        #draw rectangle    
        ax.add_patch(Rectangle((-targetWidth/2, -targetHeight/2), targetWidth, targetHeight, fill=False, ls='dashed'))
    
        #draw colorbar    
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(p, cax=cax)
        if autoSaveFigures:
            savefig(scanName + " Uniformity Plot")
    
#==============================================================================
#     #Surface Plot
#==============================================================================
    if surfacePlot:
        fig = plt.figure(num = "Surface Plot (" + scanName + ")")
        ax = fig.add_subplot(111)    
        ax.set_title(scanName + ' Surface Plot')    
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap = cmap, linewidth=0)
        ax.set_zlim(0,1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf)
        plt.axis('equal', adjustable='box')
        if autoSaveFigures:    
            savefig(scanName + " surface plot")
    
#==============================================================================
#     #Long Axis Plot
#==============================================================================
    if longAxisPlot:  
        if csvExport:
            with open(scanName + " long axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(xi[0,:], longAxisPower)
                for row in rows:
                    writer.writerow(row)
            output.close()
        fig = plt.figure(num = "Long Axis Plot (" + scanName + ")")
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Long Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(np.arange(0., 1.+1./gridResolution, 1./gridResolution))
        ax.grid(True, which='both')
        plt.axis([int(np.min(xi)), int(np.max(xi)), 0, np.max(longAxisPower)*1.1])  
        ax.plot(xi[0,:], longAxisPower)
        if autoSaveFigures:
            savefig(scanName + " long axis plot")    
        
#==============================================================================
#     #Short Axis Plot
#==============================================================================
    if shortAxisPlot:
        if csvExport:
            with open(scanName + " short axis data.csv", 'wb') as output:
                writer = csv.writer(output)
                rows = zip(yi[:,0], shortAxisPower)
                for row in rows:
                    writer.writerow(row)
            output.close()
        fig = plt.figure(num = "Short Axis Plot (" + scanName + ")")
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Short Axis Scan')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        plt.gca().set_xticks(yTicks)
        ax.grid(True, which='both')
        plt.axis([int(np.min(yi)), int(np.max(yi)), 0, np.max(shortAxisPower)*1.1])
        ax.plot(yi[:,0], shortAxisPower)
        if autoSaveFigures:
            savefig(scanName + " short axis plot")
    
    
#==============================================================================
#     Generate Contour and Uniformity Plot Overlapped (Special Request)
#==============================================================================
    if contourAndUniformityPlot:
        fig = plt.figure(num = "Contour and Uniformity Plot (" + scanName + ")")
        divider = make_axes_locatable(plt.gca())    
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Contour And Unifromity Plot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_xticks(xTicks)
        plt.gca().set_yticks(yTicks)
        plt.pcolormesh(xi, yi, zi, cmap = cmap)
        plt.axis([np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
        #draw rectangle      
        ax.add_patch(Rectangle((-targetWidth/2, -targetHeight/2), targetWidth, targetHeight, fill=False, ls='dashed'))
        
        uniformity, samples = calcUniformity(0)    
        labels = np.around(samples[:,2], decimals=3)
        samplePoints = []
        for label, x, y, in zip(labels, samples[:,0], samples[:,1]):
            #generate labels        
            plt.annotate(
                '{0:3.1%}'.format(label),
                xy = (x, y), xytext = (x, y),
                textcoords = 'offset points', ha = 'center', va = 'center',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 1),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))
            #generate circle
            circle = Circle((x, y), aperture/2)
            samplePoints.append(circle)
        color = samples[:,2]
        p = PatchCollection(samplePoints, cmap = cm.Greys, lw=0)
        p.set_clim([0, 1])
        p.set_array(np.array(color))
        ax.add_collection(p)
            
        #drawing colorbar for contour
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(cax = cax)
        
        #drawing colorbar for uniformity
        cax = divider.append_axes("left", size="5%", pad=0.5)
        cb = plt.colorbar(p, cax=cax)
        cb.ax.yaxis.set_ticks_position('left')
        if autoSaveFigures:
            savefig(scanName + "contour and uniformity plot")
#==============================================================================
#     Generate Interpoalted plot that goes corner to corner
#
#     Currently can only accomidate square profile lamps            
#==============================================================================
    if diagonalAxisPlot:
        #induce a change in coordinates
        #determine rotation angle (radians)
        #rotationAngle = np.arctan(targetHeight/targetWidth)
        #xiPrime = np.cos(rotationAngle)*xi + np.sin(rotationAngle)*yi
        #yiPrime = -np.sin(rotationAngle)*xi + np.cos(rotationAngle)*xi
        xDiagStartIndex = find_nearest_index(xi[0,:],-targetWidth * 1.1 / 2)
        yDiagStartIndex = find_nearest_index(yi[:,0],-targetHeight * 1.1 / 2)
        
        xDiagEndIndex = find_nearest_index(xi[0,:], targetWidth * 1.1 / 2)
        Xidx, Yidx = xDiagStartIndex, yDiagStartIndex
        diagPower = []    
        while Xidx < xDiagEndIndex:
            diagPower.append(zi[Yidx,Xidx])
            Yidx += 1
            diagPower.append(zi[Yidx,Xidx])
            Xidx += 1
            fig = plt.figure(num = "Short Axis Plot (" + scanName + ")")
        
        ax = fig.add_subplot(111)
        ax.set_title(scanName + ' Diagonal Axis Scan')
        #ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Normalized Output')
        plt.gca().set_xticks(yTicks)
        ax.grid(True, which='both')
        #plt.axis([int(np.min(daigPower)), int(np.max(diagPower)), 0, np.max(shortAxisPower)*1.1])
        ax.plot(diagPower)
        if autoSaveFigures:
            savefig(scanName + " diagonal axis plot")
    
    plt.show(True)
    
#functions
    
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

def calcUniformity(boxSizeRatio):
    xS = (targetWidth*(1-boxSizeRatio)-aperture)/2
    yS = (targetHeight*(1-boxSizeRatio)-aperture)/2
       
    samplePointsIndexes = np.mat([
        [find_nearest_index(xi[0,:], -xS), find_nearest_index(yi[:,0], -yS)],
        [find_nearest_index(xi[0,:], 0), find_nearest_index(yi[:,0], -yS)],
        [find_nearest_index(xi[0,:], xS), find_nearest_index(yi[:,0], -yS)],
        [find_nearest_index(xi[0,:], -xS), find_nearest_index(yi[:,0], 0)],
        [find_nearest_index(xi[0,:], 0), find_nearest_index(yi[:,0], 0)],
        [find_nearest_index(xi[0,:], xS), find_nearest_index(yi[:,0], 0)],
        [find_nearest_index(xi[0,:], -xS), find_nearest_index(yi[:,0], yS)],
        [find_nearest_index(xi[0,:], 0), find_nearest_index(yi[:,0], yS)],
        [find_nearest_index(xi[0,:], xS), find_nearest_index(yi[:,0], yS)]])
    
    samples = np.empty([0,3])
    for n in range(len(samplePointsIndexes)):  
        newRow  = np.array([xi[0, samplePointsIndexes[n, 0]], 
                            yi[samplePointsIndexes[n, 1], 0], 
                            zi.data[samplePointsIndexes[n, 1], 
                            samplePointsIndexes[n, 0]]])
        samples = np.vstack([samples, newRow])    
    uniformity = (np.max(samples[:,2])-np.min(samples[:,2]))/np.max(samples[:,2])
    return uniformity, samples


def parseRowRawData(fileName):
    #read the CSV file
    my_data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
    
    #parse the CSV file
    x = my_data[:,0]
    y = my_data[:,1]
    z = my_data[:,2]

    return x, y, z

def gridInterpolation(x, y, z, pixelPitch):
    xi = np.arange(min(x), max(x)+pixelPitch, pixelPitch)
    yi = np.arange(min(y), max(y)+pixelPitch, pixelPitch)
    #Generating a regular grid to interpolate the data    
    xi, yi = np.meshgrid(xi, yi)
    #interpolating using delaunay triangularization/natural neighbor
    zi = mlab.griddata(x,y,z,xi,yi, interp='linear')
    #handle NaN issues
    zi = np.nan_to_num(zi)
    
    #Scaling Z values should normalPlot be set to True
    zi = zi / np.max(zi)
    return xi, yi, zi

def centerOrigin(xi, yi, zi, powerBoundaryPercentage):
   ### Center Finding Portion ###
    #generating new variables to facilitate edge-finding
    X, Y, Z = sorted(np.unique(xi)), sorted(np.unique(yi)), zi.data
    
    #finding Y value that corresponds to highest power readings for all X
    highestPower = 0
    for n in range(len(Y)):
        totalPower = sum(Z[n,:])
        if totalPower > highestPower:
            highestPower = totalPower
            longAxisPower = Z[n,:]
    
    #setting the value that will be used to determine the cutoff...
    boundaryValue = max(longAxisPower)*powerBoundaryPercentage
    
    #finding x Boundaries
    xBoundary1, xBoundary2 = find_boundaries(X, longAxisPower, boundaryValue)
    
    #determining middle of lamp along X axis
    midPointX = np.average([xBoundary1, xBoundary2])
    xi = xi - midPointX
    
    shortAxisPower = Z[:,find_nearest_index(xi[0],0)]
    yBoundary1, yBoundary2 = find_boundaries(Y, shortAxisPower, boundaryValue)
    
    midPointY = np.average([yBoundary1, yBoundary2])
    yi = yi - midPointY
    ### End Center Finding ### 
    return xi, yi, midPointX, midPointY, longAxisPower, shortAxisPower

def parseArguments():
       
    defaultFileName = "sampleData.csv"
    #used in the naming of the plots    
    defaultScanName = 'Sample Data'
    
    #Dual Axis Plots
    defaultContourPlot = True
    defaultTopographicalPlot = True    
    defaultSurfacePlot = False
    defaultUniformityPlot = False    

    contourAndUniformityPlot = False 
    defaultUniformityVsBoxSizeRatioPlot = False    

    #Single Axis Plots
    defaultLongAxisPlot = False
    defaultShortAxisPlot = False
    defaultDiagonalAxisPlot = False   

    #determines grid line spacing on topographical plot
    defaultGridResolution = 10
    #number of topographical plot lines to display
    defaultTopographicalResolution = 20

    
    #lamp boundary to render
    defaultPowerBoundaryPercentage = .80
    
    #lamp profile to evaluate    
    defaultTargetWidth, defaultTargetHeight = 100, 100

    #default to 1mm, can be lowered for higher resolution plots
    defaultPixelPitch = .2
    
    #used in the uniformity calculation
    defaultAperture = 12.5
    
    #export CSV file contents
    defaultCSVExport = False
    
    #automatically save figures
    defaultAutoSaveFigures = False

    #for more info on argparse see: http://pymotw.com/2/argparse/
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',             action='store', dest='fileName', default=defaultFileName, help='The file name of the csv file used for post processing')
    parser.add_argument('-n', '--name',             action='store', dest='scanName', default=defaultScanName, help='Naming the scan, the name will be applied to plot titles')    
    parser.add_argument('-g', '--grid',             action='store', dest='gridResolution', default=defaultGridResolution, help='the spacing between grid lines', type=int)
    parser.add_argument('-l', '--lines',            action='store', dest='topographicalResolution', default=defaultTopographicalResolution, help='number of contour lines to draw on the topographical plot', type=int)
    parser.add_argument('-b', '--boundary',         action='store', dest='powerBoundaryPercentage', default=defaultPowerBoundaryPercentage, help='Percentage of max power to draw region relative to lamp profile', type=float)
    parser.add_argument('-p', '--pitch',            action='store', dest='pixelPitch', default=defaultPixelPitch, help='Distance between interpolated points', type=float)
    parser.add_argument('-w', '--width',            action='store', dest='targetWidth', default=defaultTargetWidth, help='Desired Light Profile Width (mm)', type=int)
    parser.add_argument('-H', '--height',           action='store', dest='targetHeight', default=defaultTargetHeight, help='Desired Light Profile Height (mm)', type=int)
    parser.add_argument('-a', '--aperture',         action='store', dest='aperture', default=defaultAperture, help='Diameter of aperture used on sensor', type=float)
    #boolean conditions        
    parser.add_argument('-t', '--topographical',    action='store_true', dest='topographicalPlot', default=defaultTopographicalPlot, help='Generate Topograhical Plot')
    parser.add_argument('-c', '--contour',          action='store_true', dest='contourPlot', default=defaultContourPlot, help='Generate Contour Plot')
    parser.add_argument('-L', "--long",             action='store_true', dest='longAxisPlot', default=defaultLongAxisPlot, help='Generate Long Axis Plot')
    parser.add_argument('-S', "--short",            action='store_true', dest='shortAxisPlot', default=defaultShortAxisPlot, help='Generate Short Axis Plot')
    parser.add_argument('-D', "--diagonal",         action='store_true', dest='diagAxisPlot', default=defaultDiagonalAxisPlot, help='Generate Diagonal Plot')    
    parser.add_argument('-s', '--surface',          action='store_true', dest='surfacePlot', default=defaultSurfacePlot, help='Geenrate Surface Plot')
    parser.add_argument('-u', '--uniformity',       action='store_true', dest='uniformityPlot', default=defaultUniformityPlot, help='Generate Uniformity Plot')
    parser.add_argument('-U', '--uniformityCheck',  action='store_true', dest='uniformityVsBoxSizeRatioPlot', default=defaultUniformityVsBoxSizeRatioPlot, help='Plot to ensure the sample points used for uniformity calculation are accurate')
    parser.add_argument('-A', '--autoSave',         action='store_true', dest='autoSave', default=defaultAutoSaveFigures, help='Auto Save Figures')
    parser.add_argument('-e', '--exportcsv',        action='store_true', dest='csvExport', default=defaultCSVExport, help='Export CSV Data from interpolated long and short axis scans')        
    
    args = parser.parse_args()
    
    argument = {}
    argument['fileName'] = args.fileName
    argument['gridResolution'] = args.gridResolution
    argument['topographicalResolution'] = args.topographicalResolution
    argument['scanName'] = args.scanName
    argument['powerBoundaryPercentage'] = args.powerBoundaryPercentage
    argument['pixelPitch'] = args.pixelPitch
    argument['targetWidth'] = args.targetWidth
    argument['targetHeight'] = args.targetHeight
    argument['topographicalPlot'] = args.topographicalPlot
    argument['contourPlot'] = args.contourPlot
    argument['longAxisPlot'] = args.longAxisPlot
    argument['shortAxisPlot'] = args.shortAxisPlot
    argument['diagonalAxisPlot'] = args.diagAxisPlot
    argument['surfacePlot'] = args.surfacePlot
    argument['uniformityPlot'] = args.uniformityPlot
    argument['uniformityVsBoxSizeRatioPlot'] = args.uniformityVsBoxSizeRatioPlot
    argument['aperture'] = args.aperture
    argument['autoSaveFigures'] = args.autoSave
    argument['csvExport'] = args.csvExport
    
    return argument

if __name__ == "__main__":
    main()