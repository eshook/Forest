"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Load forest
from forest import *

# Switch Engine to GPU
print("Original Engine",Config.engine)
Config.engine = pass_engine
Config.engine = cuda_engine
print("Running Engine",Config.engine)

MATRIX_SIZE = 100

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(initialize_grid.size(MATRIX_SIZE) == empty_grid.size(MATRIX_SIZE) == initialize_kernel < local_diffusion == non_local_diffusion > AGStore.file("output.tif"))


