"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Import forest code
from .core import *
from .bobs import *
from .engines import *
from .primitives import *
from .patterns import *

# Reset system paths to include the modules
# This is necessary for multiprocessing to work properly
# Otherwise the whole system will break (especially using Windows)

import sys,os

cwd = os.getcwd()
# Add current working directory
sys.path.append(cwd)

# Add subdirectories in forest
# Not including core, because that should automatically be imported
for subdir in ["bobs","engines","primitives","patterns"]:
    sys.path.append(os.path.join(cwd,subdir))


