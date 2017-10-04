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

    def __call__(self, points, others):
        partialSTC = STCube(points.y, points.x, points.h, points.w, points.s, points.d)
        partialSTC.nrows = int(points.h/others[0])
        partialSTC.ncols = int(points.w/others[0])
        partialSTC.cellwidth = others[0]
        partialSTC.cellheight = points.d/len(points.timelist)
        partialSTC.setdata()
        partialSTC.timelist = points.timelist

        return [points, partialSTC, others[1], others[2]]

setUp = rearrangeData("Set Up Data")

#Uses the STCube BOB data structure
class partialSTKDE(Primitive):

    def __call__(self, points, partialSTC, searchRadius, timeGap):
        #We need a better way to convert arrays and vectors into point type lists
        pointList = points.data[0]
        pointValues = points.data[1]
        pointTree = sp.KDTree(pointList)
        
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


##class partialSTKDEVector(Primitive):
##
##    def __call__(self, partialSTC, others):
##        searchRadius = others[0]
##        timeGap = others[1]
##        points = others[2]
##        
##        partialSTC = partialSTC[0]
##
##        for row in range(len(partialSTC.data)):
##            for column in range(len(partialSTC.data[row])):
##                yVal, xVal = partialSTC.findCellCenter(row, column)  
##
##
##                pointList = []
##                for point in points.data:
##                    distance = ((points.data[point]["geometry"]["coordinates"][0]-xVal)**2+(points.data[point]["geometry"]["coordinates"][1]-yVal)**2)**0.5
##
##                    if distance <= searchRadius:
##                        pointList.append([points.data[point], distance])
##
##                for time in range(len(partialSTC.data[row][column])):
##                    
##                
##                
##                partialSTC.data[row][column][timeInt] += points.data[point]["attributes"] * (1 - distance/searchRadius)
##                    
##
##
##
##
##'''
##                for timeInt in range(len(partialSTC.data[row][column])):
##                    for point in points.data:
##                        #Is the distance calculated as if the time was the same? Or is is a three dimensional distance calculation?
##                        if points.data[point]["geometry"]["coordinates"][2] in range(timeInt - timeGap, timeInt+1):
##                            distance = ((points.data[point]["geometry"]["coordinates"][0]-xVal)**2+(points.data[point]["geometry"]["coordinates"][1]-yVal)**2)**0.5
##                            
##                            if distance <= searchRadius:
##                                partialSTC.data[row][column][timeInt] += points.data[point]["attributes"] * (1 - distance/searchRadius)
##                   
##'''
##
##        return [partialSTC]
##
##STKDEVector = partialSTKDEVector("Partial KDE")



class kernelDensityEstimation(Pattern):

    def __call__(self, dataFileName, cellSize, searchRadius, timeGap):

        print("Running", self.__class__.__name__)
        Config.inputs = [dataFileName, [cellSize, searchRadius, timeGap]]
        output = run_primitive(CsvNewRead == setUp < STKDE > multiGeoWriter)
 
        return output


STKernelDensity = kernelDensityEstimation("Kernel Density Estimation")


