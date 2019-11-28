"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

try:
    from .Engines import *
except:
    print(" [ INFORMATION ] Engines failed to load. Likely missing libraries.") 
    pass
    
try:
    from .CUDAEngine import *
except:
    print(" [ INFORMATION ] CUDA Engine failed to load. Likely missing CUDA libraries.")
    pass

