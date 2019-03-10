"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from forest import *
import unittest

# Test forest/bobs/Bob.py

def maketiles(nfiles,nrows,ncols):
   for f_i in range(nfiles):
        f = open("unittests/tmp_raster"+str(f_i)+".asc","w")
        f.write("ncols "+str(ncols)+"\n")
        f.write("nrows "+str(nrows)+"\n")
        f.write("xllcorner 0.0\n")
        f.write("yllcorner 0.0\n")
        f.write("cellsize 1.0\n")
        f.write("NODATA_value -999\n")
        
        for i in range(nrows):
            for j in range(ncols):
                f.write(str(i+j+f_i)+" ")
            f.write("\n")

        f.close()

#maketiles(nfiles,nrows,ncols)


class TestIO(unittest.TestCase):
    def setUp(self):
        nfiles = 3 
        nrows = 13
        ncols = 13

        maketiles(nfiles,nrows,ncols)
        
    def test_io(self):
        b1 = Raster()
        self.assertEqual(b1.y,0)

        
test_IO_suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)
