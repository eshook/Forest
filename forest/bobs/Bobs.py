"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Luyi Hunter, chen3461@umn.edu; Xinran Duan, duanx138@umn.edu)
@contributors: <Contribute and add your name here!>
"""
import numpy as np

from .Bob import *

'''
# TODO
1. Should Bobs support the __geointerface__? https://gist.github.com/sgillies/2217756
   https://pysal.readthedocs.io/en/latest/users/tutorials/shapely.html
   I think so. How best to do it? Only works for Vector so maybe it should be in the vector one only?
'''


# Raster Layer Bob
class Raster(Bob):
    
    def __init__(self, y = 0, x = 0, h = 10, w = 10, s = 0, d = 0, nrows = 10, ncols = 10, cellsize = 1, filename = None, nodatavalue = None):
        
        # Call the __init__ for Bob        
        super(Raster, self).__init__(y, x, h, w, s, d)

        # Set the number of rows and columns
        self.nrows = nrows
        self.ncols = ncols

        # Set the cellsize
        self.cellsize = cellsize
        
        # FIXME: FIXED DATATYPE RIGHT NOW
        self.datatype = "Float" 
        
        # FIXME: Make this optional in the future (with a flag)
        # Create a zero raster array
        self.data = np.zeros((self.nrows,self.ncols))
        
        # FIXME: Sanity check h/w with nrows/ncols * cellsize
        # either reset Bob or print warning (flag?)

    def get_data(self, r, c, rh, cw):
        # Remember, the array is upside down in GIS
        # So when we take a slice, we must do it from the bottom
        # which is why we are subtracting r/rh from nrows
        return self.data[self.nrows-(r+rh):self.nrows-r,c:c+cw]

    # Generator that loops over the raster data
    # and returns the r, c for each element  
    def iterrc(self):
        for r in range(self.nrows):
            for c in range(self.ncols):
                yield r,c
    
    # Generator that loops over the raster data
    # and returns the r,c for each element BUT DOES NOT INCLUDE
    # a buffered area along the edge of the raster defined by buffersize
    def iterrcbuffer(self,buffersize):
        for r in range(buffersize,self.nrows-buffersize):
                for c in range(buffersize,self.ncols-buffersize):
                    yield r,c

    #Obtains the spatial coordinates given
    #a row and column number, and the size of the raster cells
    #Mckenzie Ebert
    def findCellCenter(self, row, column):
        y = self.y + (row*self.cellsize + self.cellsize/2)
        x = self.x + (column*self.cellsize + self.cellsize/2)
        return y, x
    
# Vector Layer Bob
class Vector(Bob):        
    def __init__(self,y = 0, x = 0, h = 10, w = 10, s = 0, d = 0):
        super(Vector,self).__init__(y,x,h,w,s,d)
        
        self.sr = None
        self.geom_types = [] # The geometry types the VBob holds.
        
    
    def getFeature(self,fid):
        return self.data[fid]
    
    def createFeature(self,feature):
        self.data[len(self.data)] = feature
    
    """returns a string displaying the geometry type of a feature defined by its FID in the layer.
    If no fid is specified, it returns the geometry type of the first feature in the layer."""
    def getGeomType(self,fid=None):
        if fid == None:
            return self.geom_types
        else:
            return self.data[fid]["geometry"]["type"]
    
    def getSpatialRef(self):
        return self.sr
    
    def getFeatureCount(self):
        return len(self.data)

    #For use in creating KDTrees from vector type data
    #This needs to be optimized, or a better way of converting needs to be looked
    #into, as this currently eats up processing times. Mckenzie Ebert
    def getPointListVals(self, attributeName):
        #Set up a check that makes sure self.geom_types only contains points
        pointList = []
        pointValues = []
        for point in self.data:
            pointCoords = self.data[point]["geometry"]["coordinates"]
            pointVal = self.data[point]["attributes"][attributeName]

            pointList.append(pointCoords)
            pointValues.append(pointVal)

        #Returns a list with point coordinates and another list
        #with corresponding indexes that hold point values for the specified attribute
        return pointList, pointValues

    #Used when getting Space-Time Point List values
    def getSTCPoints(self, attributeName):
        pointList = []
        pointValues = []
        timeList = []

        for point in self.data:
            pointCoords = [self.data[point]["geometry"]["coordinates"][0], self.data[point]["geometry"]["coordinates"][1]]
            #There needs to be a global name for the time attribute, as well as a global format. Maybe something to look into for I/O?
            pointVal = [self.data[point]["attributes"][attributeName],self.data[point]["attributes"]["time"]] 
            if pointVal[1] not in timeList:
                timeList.append(pointVal[1])

            pointList.append(pointCoords)
            pointValues.append(pointVal)

        #Is this truly needed? Do the time values need to be in sequential order for calculations?
        timeList.sort()

        #Returns a point list containing the x and y coordinates, and
        #a pointValues list which contains the attribute value and the corresponding time value for that point
        return pointList, pointValues, timeList
  
        
# Bob to store Key-Value Pairs        
class KeyValue(Bob):
    def __init__(self, y = 0, x = 0, h = 0, w = 0, s = 0, d = 0):
        # Call the __init__ for Bob
        super(KeyValue,self).__init__(y, x, h, w, s, d)

        # Make an empty dictionary for key-values
        self.data = {}


# Bob to store a stack of rasters arranged as a space-time cube (STCube)
# Originally authored by Jacob Arndt
class STCube(Bob):
    
    def __init__(self,y = 0, x = 0, h = 10, w = 10, s = 0, d = 0):
        super(STCube,self).__init__(y,x,h,w,s,d)
        
        self.srid = None
        self.missing_value = None
        
        self.cellwidth = None
        self.cellheight = None
        self.nrows = None 
        self.ncols = None
         
        self.e = None
        self.temres = None
        self.timelist = None
        
        self.globalattributes = None
        self.variableattributes = None
        self.dimensionattributes = None

        self.data = None

        
    def setdata(self,data = None):
        if data==None:
            for time in range(len(self.timelist)):
                self.data.append(np.zeros((self.nrows,self.ncols)))
        else:    
            self.data = data
            self.nlayers = len(self.data)
            self.nrows = len(self.data[0])
            self.ncols = len(self.data[0][0])

    def getTimeList(self):
        return self.timelist

    def getPointListVals(self):
        pointList = []
        pointData = []

        for time in range(self.nlayers):
            for row in range(self.nrows):
                for column in range(self.ncols):

                    pointData.append([self.data[time][row][column], self.timelist[time]])
                    pointList.append([row, column])


    def findCellCenter(self, row, column, time=None):
        if time == None:
            y = self.y + (row*self.cellheight + self.cellheight/2)
            x = self.x + (column*self.cellheight + self.cellheight/2)
            return y, x
        else:
            y = self.y + (row*self.cellheight + self.cellheight/2)
            x = self.x + (column*self.cellheight + self.cellheight/2)
            t = self.s + (time*self.cellwidth + self.cellwidth/2)
            return y, x, t

    def iterrc(self):
        for r in range(self.nrows):
            for c in range(self.ncols):
                yield r, c
        

#2D point layer
class Point(Vector):
    def __init__(self,y = 0, x = 0, h = 10, w = 10, s = 0, d = 0):
        super(Point,self).__init__(y,x,h,w,s,d)
        #we need a seperate container for halo zones
        self.halo = []
#Spatio temporal point layer
class STPoint(Point):
    def __init__(self,y = 0, x = 0, h = 10, w = 10, s = 0, d = 0):
        super(STPoint,self).__init__(y,x,h,w,s,d)
        #we need a seperate container for halo zones
        self.halo = []        
    
