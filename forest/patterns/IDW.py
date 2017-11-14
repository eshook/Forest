'''
Inverse Distance Weighted

Mckenzie Ebert
'''

from ..bobs import *
from ..primitives import *
from ..engines import *
from .Pattern import *
import scipy.spatial as sp

#This primitive is used in reordering the input data as well as creating
#a blank raster to use for further calculations.
class setUpPrim(Primitive):

    def __call__(self, values, others):
        cellSize = others[0]
        attributeName = others[1]
        searchRadius = others[2]
        filePath = others[3]
        
        blankRaster = Raster(values.y, values.x, values.h, values.w, 0, 0, int(values.h/cellSize), int(values.w/cellSize), cellSize)
        return [blankRaster, values, attributeName, searchRadius, filePath]

setUp = setUpPrim("Set Up for next step")

#The classic version of the IDW primitive. Makes use of nested for loops to iterate
#over rows and columns of the raster, and searches through all values in the given data set.
class classicPartialIDW(Primitive):

    def __call__(self, raster, values, attributeName, filePath, power = 2):
        
        for row in range(raster.nrows):
            for column in range(raster.ncols):
                yVal, xVal = raster.findCellCenter(row, column) #Bob class function that finds the center point of a cell for the given row and column

                weightedSum = 0.0
                totalWeight = 0.0
                for dataPoint in values.data:
                    
                    distance = ((xVal-values.data[dataPoint]["geometry"]["coordinates"][0])**2+(yVal-values.data[dataPoint]["geometry"]["coordinates"][1])**2)**(0.5)
                    weight = 1.0 / distance**power

                    totalWeight += weight
                    weightedSum += weight*values.data[dataPoint]["attributes"][attributeName]
          
                raster.data[row][column] = weightedSum/totalWeight

        return [raster, values.sr, filePath]


classicIDW = classicPartialIDW("Partial IDW")


#This is the optimized form of the classic IDW calculations. It uses a built in
#generator to iterate over the raster, as well as a KDTree to find points within the searchRadius
class partialIDW(Primitive):

    #In regards to splitting values - currently the primitive is set up to take in
    #a raster that only covers a part of the total spatial area of the data. The primitive
    #will then treat this raster as the full raster, and calculations will go from there
    def __call__(self, raster, values, attrName, searchRadius, filePath, power = 2):
        #Placing the vector data into an array format in order to sort into a KDTree data structure
        pointList, pointValues = values.getPointListVals(attrName)
        pointTree = sp.cKDTree(pointList)

        #Built in row and column generator in the Raster class
        for row, column in raster.iterrc():
            xVal, yVal = raster.findCellCenter(row, column) #Bob class function that finds the center point of a cell for the given row and column

            #Obtaining points within the search radius
            pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

            #Normal IDW calculations
            weightedSum = 0.0
            totalWeight = 0.0
            for point in pointsInRadius:
                
                distance = ((xVal-pointList[point][0])**2+(yVal-pointList[point][1])**2)**(0.5)
                weight = 1.0 / distance**power

                totalWeight += weight
                weightedSum += weight*pointValues[point]

            raster.data[row][column] = weightedSum/totalWeight

        #Returns spatial reference and filePath to assist the GeoTIFF writer in creating its .tif file
        return [raster, values.sr, filePath]


partIDW = partialIDW("Partial IDW")


class IDW(Pattern):
    
    def __call__(self, dataFileName, cellSize, attributeName, searchRadius, filePath = None):
        
        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, attributeName, searchRadius, filePath]]
        output = run_primitive(ShapefileRead == setUp < partIDW > GeotiffWrite)
       
        return output

inverseDistanceWeighted = IDW("Inverse Distance Weighted Interpolation")
