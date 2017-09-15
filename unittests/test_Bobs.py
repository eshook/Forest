"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from forest import *
import unittest

# Test forest/bobs/Bob.py

class TestBobs(unittest.TestCase):
    def setUp(self):
        self.raster100 = Raster(0,0,100,100,cellsize = 1,nrows = 100, ncols = 100)
        
        # FIXME: Test for these cases too
        #self.bobneg = Bob(-1, -1, 99, 99)
        #self.bob10 = Bob(10,10,10,10)
        #self.bobrect = Bob(10,20,30,40) # y=10, x=20, h=30, w=40
        
        # FIXME: Test negative nrows, ncols, cellsize
        

    def test_basic_raster_initial_setup(self):
        self.assertEqual(self.raster100.y,0)
        self.assertEqual(self.raster100.x,0)
        self.assertEqual(self.raster100.h,100)
        self.assertEqual(self.raster100.w,100)
        self.assertEqual(self.raster100.nrows,100)
        self.assertEqual(self.raster100.ncols,100)
        self.assertEqual(self.raster100.cellsize,1)
        
        
        

# Create the TestBobs suite        
test_Bobs_suite = unittest.TestLoader().loadTestsFromTestCase(TestBobs)
