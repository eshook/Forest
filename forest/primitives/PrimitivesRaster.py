"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from .Primitive import *
from ..bobs.Bobs import *

# RFunct is an extended primitive where you can apply a function
# to the left and right raster layers or just a single raster layer.
class RFunct(Primitive):
    def __init__(self, name, function, buffersize = None):
        # Call Primitive.__init__ as the super class
        super(RFunct,self).__init__(name)
        
        # Reset the name
        self.name = "RFunct "+name
    
        # Set the function to be applied to left and right raster layers
        self.function = function

    # Define the __call__ function which will be called for all RFunct methods.
    def __call__(self, left = None, right = None):

        # Duplicate our left raster for the output # FIXME: Is this right?
        out = Raster(left.y,left.x,left.h,left.w)

        # If left and right are set, apply function to left and right
        if left != None and right != None:
            out.data = self.function(left.data,right.data)
        # If only left is set, then apply function to only left 
        elif left != None and right == None: 
            out.data = self.function(left.data)
        else: # This is a problem
            raise
        
        return out


# RMap is an extended primitive where you can apply a function
# to each element of the raster cell
# FIXME: This may need to additional work to make it truly generic
class RMap(Primitive):
    def __init__(self, name, function, buffersize = 0):
        # Call Primitive.__init__ as the super class
        super(RMap,self).__init__(name)
        
        # Reset the name
        self.name = "RMap "+name
    
        # Set the function to be applied to left and right raster layers
        self.function = function

        # Set the buffersize
        self.buffersize = buffersize
        
    # Define the __call__ function which will be called for all RMap methods.
    # FIXME: I don't think that there is a need for a 'right' here.
    def __call__(self, left = None):

        buffersize = self.buffersize # Local variable for readability
        
        # Duplicate our left raster for the output # FIXME: Is this right?
        out = Raster(left.y,left.x,left.h,left.w)

        # Loop over all the cells in the Raster Bob
        # Make sure to skip the buffered areas
        # Instead of a nested for loop, could we generate the pair of r,c
        # or use itertools.map or something to loop over the data efficiently.
        for r in range(buffersize,len(out.data)-buffersize):
            for c in range(buffersize,len(out.data[0])-buffersize):
                # Apply the function to the data in that cell
                out.data[r][c]=self.function(
                                left.data[r-buffersize:r + buffersize + 1, 
                                          c-buffersize:c + buffersize + 1])
        
        return out


# Create a number of local operations for raster data using Raster Function
        
LocalSum = RFunct("LocalSum",np.add)
LocalSub = RFunct("LocalSubtract",np.subtract)
LocalMultiply = RFunct("LocalMultiply",np.multiply)
LocalDivide = RFunct("LocalDivide",np.divide)
LocalPower = RFunct("LocalPower",np.power)
LocalMod = RFunct("LocalModulus",np.mod)
LocalRemainder = RFunct("LocalRemainder",np.remainder)
LocalFloorDivide = RFunct("LocalFloorDivide",np.floor_divide)
LocalMaximum = RFunct("LocalMaximum",np.maximum)
LocalMinimum = RFunct("LocalMinimum",np.minimum)

# Create a number of focal operations for raster data using Raster Map
FocalMean = RMap("FocalMean",np.mean,buffersize=1)
FocalMaximum = RMap("FocalMaximum",np.amax,buffersize=1)
FocalMinimum = RMap("FocalMinimum",np.amin,buffersize=1)
