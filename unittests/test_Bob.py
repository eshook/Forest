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
        self.bob100 = Bob(0,0,100,100)
        

    def test_basic_initial_setup(self):
        self.assertEqual(self.bob100.y,0)
        self.assertEqual(self.bob100.x,0)
        self.assertEqual(self.bob100.h,100)
        self.assertEqual(self.bob100.w,100)
        self.assertEqual(self.bob100.s,0)
        self.assertEqual(self.bob100.d,0)
        self.assertEqual(self.bob100.data,None)
        
    def test_basic_call(self):
        # Call the Bob, should return the data, which is None at the moment
        self.assertEqual(self.bob100(),None)
    
    def test_basic_string_output(self):
        # Call the Bob, should return the data, which is None at the moment
        self.assertEqual(str(self.bob100),"Bob (0.000000,0.000000) [100.000000,100.000000]")
        
    def test_basic_data_set(self):
        
        self.bob100.data = [[0,0],[0,0]]
        
        # Call the Bob, should return the data, which is a 2x2 array
        self.assertEqual(self.bob100(),[[0,0],[0,0]])
        

        
test_Bob_suite = unittest.TestLoader().loadTestsFromTestCase(TestBob)
