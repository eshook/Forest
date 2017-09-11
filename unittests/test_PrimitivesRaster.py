"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from forest import *
import unittest
import numpy as np

# Test forest/bobs/Bob.py

class TestPrimitivesRaster(unittest.TestCase):
    def setUp(self):
        self.raster0 = Raster(0,0,4,4,cellsize = 1,nrows = 4, ncols = 4)
        self.raster1 = Raster(0,0,4,4,cellsize = 1,nrows = 4, ncols = 4)
        
        self.raster0.data = np.zeros((4,4))
        self.raster1.data = np.ones((4,4))
        
        # FIXME: Test for these cases too
        #self.bobneg = Bob(-1, -1, 99, 99)
        #self.bob10 = Bob(10,10,10,10)
        #self.bobrect = Bob(10,20,30,40) # y=10, x=20, h=30, w=40
        

    def test_LocalSum(self):
        # Create a ones array
        ones_array = np.ones((4,4))
        
        # Add 0's and 1's
        oraster = LocalSum(self.raster0,self.raster1)
        
        # Should be 1's
        self.assertTrue((oraster.data ==ones_array).all())
        #self.assertListEqual(oraster.data,ones_array)
        
    def test_LocalMaximum(self):
        # Create a ones array
        ones_array = np.ones((4,4))
        test_array = np.ones((4,4))
        zero_array = np.zeros((4,4))
        
        test_array[0][0] = -1
        test_array[1][2] = -10
        test_array[2][1] = -50
        test_array[2][0] = -234234234        

        zero_array[0][0] = 1
        zero_array[1][2] = 1
        zero_array[2][1] = 1
        zero_array[2][0] = 1        

        self.raster0.data = zero_array
        self.raster1.data = test_array

        # Maximum of two arrays
        oraster = LocalMaximum(self.raster0,self.raster1)
                
        # Should be 1's
        self.assertTrue((oraster.data ==ones_array).all())
        #self.assertListEqual(oraster.data,ones_array)
        
    def test_LocalMinimum(self):
        # Create a ones array
        valid_array = np.zeros((4,4))
        test_array = np.ones((4,4))
        zero_array = np.zeros((4,4))
        
        test_array[0][0] = -1
        test_array[1][2] = -10
        test_array[2][1] = -50
        test_array[2][0] = -234234234        

        zero_array[0][0] = 1
        zero_array[1][2] = 1
        zero_array[2][1] = 1
        zero_array[2][0] = 1        

        valid_array[0][0] = -1
        valid_array[1][2] = -10
        valid_array[2][1] = -50
        valid_array[2][0] = -234234234        

        self.raster0.data = zero_array
        self.raster1.data = test_array

        # Minimum of two arrays
        oraster = LocalMinimum(self.raster0,self.raster1)
                
        # Should be valid array
        self.assertTrue((oraster.data ==valid_array).all())
        #self.assertListEqual(oraster.data,ones_array)
        
    
    def test_iterrc(self):
        raster0test = self.raster0
        
        # Manual list building
        manuallist = []
        for r in range(raster0test.nrows):
            for c in range(raster0test.ncols):
                manuallist.append([r,c])
        
        # iterrc generator list
        iterrclist = []
        for r,c in raster0test.iterrc():
            iterrclist.append([r,c])
                
        self.assertEqual(manuallist,iterrclist)
    
    def test_iterrcbuffer(self):

        buffersize = 1
            
        raster0test = self.raster0
        
        # Manual list building
        manuallist = []
        for r in range(buffersize,raster0test.nrows-buffersize):
            for c in range(buffersize,raster0test.ncols-buffersize):
                manuallist.append([r,c])
        
        # iterrc generator list
        iterrclist = []
        for r,c in raster0test.iterrcbuffer(buffersize):
            iterrclist.append([r,c])
                
        self.assertEqual(manuallist,iterrclist)

# Create the TestBobs suite        
test_PrimitivesRaster_suite = unittest.TestLoader().loadTestsFromTestCase(TestPrimitivesRaster)
