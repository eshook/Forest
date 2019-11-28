"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

try:
    from .Primitives import *
except:
    print(" [ INFORMATION ] Primitives failed to load. Likely missing libraries.") 
    pass

try:
    from .PrimitivesRaster import *
except:
    print(" [ INFORMATION ] PrimitivesRaster failed to load. Likely missing libraries.") 
    pass

try:
    from .PrimitivesCUDA import *
except:
    print(" [ INFORMATION ] PrimitivesCUDA failed to load. Likely missing CUDA libraries.") 
    pass

try:
    from .IO import *
except:
    print(" [ INFORMATION ] IO (Primitives) failed to load. Likely missing libraries.") 
    pass

