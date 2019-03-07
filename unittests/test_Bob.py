"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from forest import *
import unittest

# Test forest/bobs/Bob.py

class TestBob(unittest.TestCase):
    def setUp(self):
        self.bob100 = Bob(1,1,100,100)
        self.bob10010 = Bob(0,0,100,100,10,20) # t=10, d=20
        

    def test_basic_initial_setup(self):
        self.assertEqual(self.bob100.y,1)
        self.assertEqual(self.bob100.x,1)
        self.assertEqual(self.bob100.h,100)
        self.assertEqual(self.bob100.w,100)
        self.assertEqual(self.bob100.t,0)
        self.assertEqual(self.bob100.d,0)
        self.assertEqual(self.bob100.data,None)
        
    def test_basic_call(self):
        # Call the Bob, should return the data, which is None at the moment
        self.assertEqual(self.bob100(),None)
    
    def test_basic_string_output(self):
        # Call the Bob, should return the data, which is None at the moment
        self.assertEqual(str(self.bob100),"Bob (1.000000,1.000000) [100.000000,100.000000]")
        
    def test_basic_data_set(self):
        # Set a 2x2 raster array of 0 values
        self.bob100.data = [[0,0],[0,0]]
        
        # Call the Bob, should return the data, which is a 2x2 array
        self.assertEqual(self.bob100(),[[0,0],[0,0]])
        
    def test_ST_setup(self):
        self.assertEqual(self.bob10010.y,0)
        self.assertEqual(self.bob10010.x,0)
        self.assertEqual(self.bob10010.h,100)
        self.assertEqual(self.bob10010.w,100)
        self.assertEqual(self.bob10010.t,10)
        self.assertEqual(self.bob10010.d,20)
        self.assertEqual(self.bob10010.data,None)
        
        
test_Bob_suite = unittest.TestLoader().loadTestsFromTestCase(TestBob)
