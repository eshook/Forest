"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Jacob Arndt, arndt204@umn.edu; )
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
    
    def __init__(self, y = 0, x = 0, h = 10, w = 10, s = 0, d = 0, nrows = 10, ncols = 10, cellsize = 1):
        
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
        
    def setdata(self,data):
        self.data = data
        self.nlayers = len(self.data)
        self.nrows = len(self.data[0])
        self.ncols = len(self.data[0][0])

    def getTimeList(self):
        return self.timelist
        
    