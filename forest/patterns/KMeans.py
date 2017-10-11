'''
Clustering / K-Means Algorithm

Mckenzie Ebert
'''


from ..bob import *
from ..primitives import *
from .Pattern import *
from ..config import *
import random
import scipy.spatial as sp


class centerInit(Primitive):

    def __call__(self, points, others):
        k = others[0]
        filePath = [1]
        
        #Centers will hold the coordinates of the center point for each cluster, as well as a xSum, ySum, and pointCount, all of which will be used to calculate new coordinates if need be
        centers = []

        #While loop ensures that no two center points start off the same
        unique = False
        while unique == False:
            #Sets center point coordinates from a random data point
            for n in range(k):
                pointNum = random.randint(0, len(points.data)-1)
                centers.append(points.data[pointNum]["geometry"]["coordinates"])

            #Checks for uniqueness
            for point in range(len(centers)):
                if centers[point] not in centers[point+1:]:
                    unique = True
                else:
                    unique = False
                    centers = []
                    break

        #Creates 'empty' elements for storing xSum, ySum, and pointCount later on
        for index in range(len(centers)):
            centers[index]+=[0,0,0]

        return [points, centers, filePath]

kInit = centerInit("Center Point Initilization")



class findNearest(Primitive):

    def __call__(self, points = None, centers = None, oldDistance = float('inf'), filePath = None):

        newDistance = 0  

        #Current set up has each core running through every point in the data set.
        for point in points.data:
            #Finds the cluster the current data point is nearest to
            minDistance = float('inf')            
            for index in range(len(centers)):
                distance = ((centers[index][0]-points.data[point]["geometry"]["coordinates"][0])**2+(centers[index][1]-points.data[point]["geometry"]["coordinates"][1])**2)**(0.5)

                if distance < minDistance:
                    minDistance = distance
                    nearestIndex = index
                               
            #Adds the point's x-coordinate and y-coordinate to the x and y sums for the assigned center. Also increases the pointCount for that center by 1.
            centers[nearestIndex][2] += points.data[point]["geometry"]["coordinates"][0] #xSum
            centers[nearestIndex][3] += points.data[point]["geometry"]["coordinates"][1] #ySum
            centers[nearestIndex][4] += 1                                                #pointCount

            #Keeps track of overall distance to each center point
            newDistance += minDistance


        return [points, centers, newDistance, oldDistance]

nearest = findNearest("Find Nearest Points")




class checkThreshold(Primitive):

    def __call__(self, points = None, centers = None, newDistance = 0, oldDistance = float('inf'), threshold = 1e-10, filePath = None):

        oldDistance = oldDistance / len(points.data)
        newDistance = newDistance / len(points.data)


        if oldDistance - newDistance < threshold:
            #Makes the centers variable easier to read in the output. Can be altered if neccessary.
            centers = [[center[0], center[1]] for center in centers]
            return [newDistance, centers]

        else:
            #Ajusts the center point of each cluster to reflect the average x and y sum of all data points assigned to it. Then resets the sums before looping.
            for cIndex in range(len(centers)):
                if centers[cIndex][4] == 0:
                    centers[cIndex][0] = centers[cIndex][2]
                    centers[cIndex][1] = centers[cIndex][3]
                else:
                    centers[cIndex][0] = centers[cIndex][2] / centers[cIndex][4]
                    centers[cIndex][1] = centers[cIndex][3] / centers[cIndex][4]
                    
                centers[cIndex][2] = 0
                centers[cIndex][3] = 0
                centers[cIndex][4] = 0
            
            return [points, centers, newDistance]

thresholdCheck = checkThreshold("Check Threshold")


class KMeans(Pattern):

    def __call__(self, dataFileName, k, filePath = None):

        Config.inputs = [dataFileName, [k, filePath]]

        #Does not currently loop, as the algorithm requires to function.
        output = run_primitive(ShapefileRead == kInit < nearest > thresholdCheck)

        return output


##kmean = KMeans("K-Means Algorithm")

#Hardcoded a test run that implements a loop over the findNearest and checkThreshold functions
def patternTest():

    temp = kMeanRead.__call__("KMeansTest.txt", 10)
    points, k = temp[0], temp[1]
    temp = kInit(points, k)
    zones, points, centers = temp[0], temp[1], temp[2]
    temp = nearest(zones, points, centers)
    points, centers, newDistance, oldDistance, zones = temp[0], temp[1], temp[2], temp[3], temp[4]
    results = thresholdCheck(points, centers, newDistance, oldDistance, zones)

    length = len(results)

    while length > 2:
        points, centers, newDistance, oldDistance, zones = nearest(results[0], results[1], results[2], results[3])
        results = thresholdCheck(points, centers, newDistance, oldDistance, zones)

        length = len(results)

    print(results)
    




