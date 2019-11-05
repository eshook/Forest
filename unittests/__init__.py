"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Import unittests
from .test_Bob import *
from .test_Bobs import *
from .test_IO import *
from .test_PrimitivesRaster import *

# Try BMSB import with CUDA
# If it fails, then skip it
try:
    from .test_Bmsb import *
except:
    pass
