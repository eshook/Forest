'''
Uniform Kernel Density Estimation

Mckenzie Ebert
'''


from ..bob import *
from ..primitives import *
from ..engines import *
from .Pattern import *
import scipy.spatial as sp

#This primitive takes care of rearranging the data for the next primitive to use,
#as well as creating an empty raster to store data calculations in
class rearrangeData(Primitive):

    def __call__(self, points, others):

        cellSize = others[0]
        searchRadius = others[1]
        attrName = others[2]
        filePath = others[3]

        zones = Raster(points.y, points.x, points.h, points.w, 0, 0, int(points.h/cellSize), int(points.w/cellSize), cellSize)

        return [zones, points, searchRadius, attrName, filePath]

setUp = rearrangeData("Set Up Data")


#This is the classic version of the Uniform KDE calculations. It makes use
#of nested for loops to find row and column counts, and runs through all values given to it.
class classicUniformKDE(Primitive):

    def __call__(self, partialR, points, searchRadius, attrName, filePath):

        for row in range(len(partialR.data)):
            for column in range(len(partialR.data[row])):
                #This gets the spatial coordinates for calculating distance
                yVal, xVal = partialR.findCellCenter(row, column)  

                #Runs through all data points given to the primitive, testing to see if
                #they are within the give search radius
                for point in points.data:
                    distance = ((points.data[point]["geometry"]["coordinates"][0]-xVal)**2+(points.data[point]["geometry"]["coordinates"][1]-yVal)**2)**0.5
                    #Checks distance against given search radius
                    if distance <= searchRadius:
                        partialR.data[row][column] += points.data[point]["attributes"][attrName]

                #Adjusts results for given search radius
                partialR.data[row][column] = partialR.data[row][column]/searchRadius
        
        return [partialR, points.sr, filePath]

ClassicUKDE = classicUniformKDE("Partial KDE")


#This is the optimized version of the uniform KDE calculations. It makes use of KDTree
#data structures when searching for nearby points and uses a built in Raster row column generator in order
#to run through the raster cells.
class partialUniformKDE(Primitive):

    def __call__(self, partialR, points, searchRadius, attrName, filePath):
        #Puts the data points into array format, for easier access and use in the KDTree
        pointList, pointValues = points.getPointListVals(attrName)
        pointTree = sp.cKDTree(pointList)

        #Built in Raster class row column generator
        for row, column in partialR.iterrc():
            #Gets the spatial coordinates for the middle of the current cell
            yVal, xVal = partialR.findCellCenter(row, column)  
            #Finds all of the points within a certain distance of the current point
            pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

            #Does the KDE calculations on the found points
            for point in pointsInRadius:
                distance = ((pointList[point][0]-xVal)**2+(pointList[point][1]-yVal)**2)**0.5
                partialR.data[row][column] += pointValues[point]

            #Adjusts the results for the search radius
            partialR.data[row][column] = partialR.data[row][column]/searchRadius

        #Returns the spatial reference and file path for the GeoTIFF primitive to use
        return [partialR, points.sr, filePath]

uKDE = partialUniformKDE("Partial KDE")



class uniformKDE(Pattern):

    def __call__(self, dataFileName, cellSize, searchRadius, attrName, filePath = None):

        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, searchRadius, attrName, filePath]]
        output = run_primitive(ShapefileRead == setUp < uKDE > GeotiffWrite) 

        return output


uniformKernelDensity = uniformKDE("Uniform Kernel Density Estimation")
