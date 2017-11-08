'''
Space-Time Kernel Density Estimation

Mckenzie Ebert
'''


from ..bobs import *
from ..primitives import *
from ..engines import *
from .Pattern import *
import scipy.spatial as sp

#Takes the Vector BOB from the ShapefileRead Prim and makes it into a STCube BOB
class setUpSTPrim(Primitive):

    def __call__(self, points, others = None):
        #Changing the Vector into a STCube
        results = VectorToSTCube.__call__(points, others[3], others)
        points = results[0]
        others = results[1]
        partialSTC = STCube(points.y, points.x, points.h, points.w, points.s, points.d)
        partialSTC.nrows = int(points.h/others[0])
        partialSTC.ncols = int(points.w/others[0])
        partialSTC.cellheight = others[0]
        
        return [points, partialSTC, others[1], others[2], others[3], others[4]]

setUpST = setUpSTPrim("Set Up Data")



#Classic version of Space-Time KDE
class classicPartialSTKDE(Primitive):

    def __call__(self, points, partialSTC, searchRadius, timeGap):

        for time in range(len(partailSTC.data)):
            for row in range(len(partialSTC.data[time])):
                for column in range(len(partialSTC.data[time][row])):
                    yVal, xVal = partialSTC.findCellCenter(row, column)  

                    for point in points.data:
                        timeStamp = points.data[point]["attributes"]["time"]
                        if timeStamp in range(time - timeGap, time+1):
                            distance = ((points.data[point]["geometry"]["coordinates"][0]-xVal)**2+(points.data[point]["geometry"]["coordinates"][1]-yVal)**2)**0.5
                            
                            if distance <= searchRadius:
                                partialSTC.data[timeInt][row][column] += points.data[point]["attributes"][attrName] * (1 - distance/searchRadius)
                   
        return [partialSTC]

classicSTKDE = classicPartialSTKDE("Partial KDE")



#Takes in points stored in the STCube BOB data structure
class partialSTKDE(Primitive):
    #It's assumed that each core is only given a partial STCube to analyze - partialSTC would not have the same dimensions in real run through
    def __call__(self, points, partialSTC, searchRadius, timeGap, attrName, filePath = None):
        #Puts the data into a form that cKDTree can understand - look into a faster and more efficient way to do so
        pointList, pointValues = points.getPointListVals()
        pointTree = sp.cKDTree(pointList)

        #Iterates over all rows, columns, and layers in the partial STCube
        for row, column in partialSTC.iterrc():
            #Finds the actual coordinates of the cell
            yVal, xVal = partialSTC.findCellCenter(row, column)
            #Finds points near where the calculations should be done
            pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

            for time in range(len(points.timelist)):
                timeEnd = points.timelist[time]+timeGap

                for point in pointsInRadius:
                    #Checks if the time is within the given range of values
                    if (pointValues[point][1] >= time) and (pointValues[point][1] <= timeEnd):
                        distance = ((pointList[point][0]-xVal)**2+(pointList[point][1]-yVal)**2)**0.5
                        partialSTC.data[time][row][column] += pointValues[point][1] * (1 - distance/searchRadius)

                #Adjusts the data
                partialSTC.data[time][row][column] = partialSTC.data[time][row][column]/searchRadius

        return [partialSTC, points.sr, attrName, filePath]

STKDE = partialSTKDE("Partial Space-Time KDE")


class kernelDensityEstimation(Pattern):

    def __call__(self, dataFileName, cellSize, searchRadius, timeGap, attrName, filePath = None):

        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, searchRadius, timeGap, filePath]]
        output = run_primitive(ShapefileRead == setUpST < STKDE)
 
        return output


STKernelDensity = kernelDensityEstimation("Kernel Density Estimation")


