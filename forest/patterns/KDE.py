'''
Kernel Density Estimation

Mckenzie Ebert
'''

from ..bob import *
from ..primitives import *
from ..engines import *
from .Pattern import *
import scipy.spatial as sp

#This primitive is used to set-up the data coming in from the input data files
#and to create the raster needed for further computations
class rearrangeData(Primitive):

    def __call__(self, points, others):
        cellsize = others[0]
        searchRadius = others[1]
        attrName = others[2]
        filePath = others[3]

        zones = Raster(points.y, points.x, points.h, points.w, 0, 0, int(points.h/cellsize), int(points.w/cellsize), cellsize)
        return [zones, points, searchRadius, attrName, filePath]

KDEsetUp = rearrangeData("Set Up Data")

#The "classic" method of finding KDE. Altered later optimization purposes, the original
#primitive runs over the entire dataset given to it when searching for nearby points.
#Also uses nested for loops to obtain row and column numbers, rather than a generator
class classicPartialKDE(Primitive):

    def __call__(self, partialR, points, searchRadius, attrName, filePath):
      
        for row in range(len(partialR.data)):
            for column in range(len(partialR.data[row])):
                #Gets the actual spatial coordinates for use in distance calculations
                yVal, xVal = partialR.findCellCenter(row, column)

                 #Runs over all data points, and tests to see if they are within the searchRadius (bottleneck)
                for point in points.data:
                    distance = ((points.data[point]["geometry"]["coordinates"][0]-xVal)**2+(points.data[point]["geometry"]["coordinates"][1]-yVal)**2)**0.5
                    
                    if distance <= searchRadius:
                        partialR.data[row][column] += points.data[point]["attributes"][attrName] * (1 - distance/searchRadius)

                #Adjusts values
                partialR.data[row][column] = partialR.data[row][column]/searchRadius
                                                                                        
        return [partialR, points.sr, filePath]

classicKDE = classicPartialKDE("Classic Partial KDE")


#This is the optimized version of the classic KDE calculations. It makes use of both
#scipy.spatial.KDTree and a Raster type generator. Using the KDTree method
#query_ball_point, the method only runs over those points which are within the
#searchRadius, rather than the entire data set given to it. The current Raster method
#iterrc() still needs to be optimized, but this will allow all patterns to be updated
#at once.
class partialKDE(Primitive):

    def __call__(self, partialR, points, searchRadius, attrName, filePath):
        pointList, pointValues = points.getPointListVals(attrName) #Places coordinates and their corresponding values into arrays, to be given to the KDTree for sorting
        pointTree = sp.cKDTree(pointList)

        
        for row, column in partialR.iterrc():
            #Finds the spatial coordinate of the point, to be used in distance calculations
            yVal, xVal = partialR.findCellCenter(row, column)

            #Gets all points within the searchRadius
            pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

            #Iterates over the points found in the KDTree
            for point in pointsInRadius:
                distance = ((pointList[point][0]-xVal)**2+(pointList[point][1]-yVal)**2)**0.5
                partialR.data[row][column] += pointValues[point] * (1 - distance/searchRadius)

            #Adjusts data for searchRadius size
            partialR.data[row][column] = partialR.data[row][column]/searchRadius

        #Returns the spatial reference and filePath for the GeoTIFF writer primitive to use
        return [partialR, points.sr, filePath]

KDE = partialKDE("Partial KDE")


class kernelDensityEstimation(Pattern):

    def __call__(self, dataFileName, cellSize, searchRadius, attrName, filePath = None):

        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, searchRadius, attrName, filePath]]
        output = run_primitive(ShapefileRead == KDEsetUp < KDE > GeotiffWrite) 

        return output

kernelDensity = kernelDensityEstimation("Kernel Density Estimation")





