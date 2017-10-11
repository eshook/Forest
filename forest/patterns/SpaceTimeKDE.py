'''
Space-Time Kernel Density Estimation

Mckenzie Ebert
'''


from ..bob import *
from ..primitives import *
from .Pattern import *
from ..config import *
import scipy.spatial as sp


class rearrangeData(Primitive):

    def __call__(self, points, others = None):
        partialSTC = STCube(points.y, points.x, points.h, points.w, points.s, points.d)
        partialSTC.nrows = int(points.h/others[0])
        partialSTC.ncols = int(points.w/others[0])
        partialSTC.cellheight = others[0]
        #Only applicable if assuming points is a STCube, not a vector
        #partialSTC.cellheight = points.d/len(points.timelist)
        #partialSTC.timelist = points.timelist
        #partialSTC.setdata()
        
        return [points, partialSTC, others[1], others[2], others[3], others[4]]

setUp = rearrangeData("Set Up Data")



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



#Assumes points is a STCube BOB data structure
class partialSTKDE(Primitive):

    def __call__(self, points, partialSTC, searchRadius, timeGap, filePath):
        #We need a better way to convert arrays and vectors into point type lists
        pointList = points.data[0]
        pointValues = points.data[1]
        pointTree = sp.cKDTree(pointList)
        
        for row, column in partialSTC.iterc():
            yVal, xVal = partialSTC.findCellCenter(row, column)

            pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

            for time in range(len(points.timelist)):
                timeEnd = points.timelist[time]+timeGap

                for point in pointsInRadius:
                    #This is not an inclusive range at the moment, but we need to know the time interval/increasing factor to make it inclusive
                    if pointData[point][0] in range(time, timeEnd):
                        distance = ((pointList[point][0]-xVal)**2+(pointList[point][1]-yVal)**2)**0.5
                        partialSTC.data[time][row][column] += pointValues[point][1] * (1 - distance/searchRadius)

                partialSTC.data[time][row][column] = partialSTC.data[time][row][column]/searchRadius

        return [partialSTC, points.sr, filePath]

STKDE = partialSTKDE()


#Assumes vector data stucture for points
class partialSTKDEVector(Primitive):

    def __call__(self, points, partialSTC, searchRadius, timeGap, filePath):
        pointList, pointValues, timeList = points.getSTCPoints(attrName)
        partialSTC.timelist = timeList
        partialSTC.setdata()
        pointTree = sp.cKDTree(pointList)


        for row, column in partialSTC.iterrc():
                yVal, xVal = partialSTC.findCellCenter(row, column)

                pointsInRadius = pointTree.query_ball_point([xVal, yVal], searchRadius)

                for time in range(len(partialSTC.timelist)):
                timeEnd = partialSTC.timelist[time]+timeGap

                for point in pointsInRadius:
                    #This is not an inclusive range at the moment, but we need to know the time interval/increasing factor to make it inclusive
                    if pointData[point][0] in range(time, timeEnd):
                        distance = ((pointList[point][0]-xVal)**2+(pointList[point][1]-yVal)**2)**0.5
                        partialSTC.data[time][row][column] += pointValues[point][1] * (1 - distance/searchRadius)

                partialSTC.data[time][row][column] = partialSTC.data[time][row][column]/searchRadius
                    

        return [partialSTC, points.sr, filePath]

STKDEVector = partialSTKDEVector("Partial KDE")



class kernelDensityEstimation(Pattern):

    def __call__(self, dataFileName, cellSize, searchRadius, timeGap, filePath = None):

        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, searchRadius, timeGap, filePath]]
        output = run_primitive(ShapefileRead == setUp < STKDEVector > multiGeoWriter)
 
        return output


STKernelDensity = kernelDensityEstimation("Kernel Density Estimation")


