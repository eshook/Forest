"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Load forest
from forest import *

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# Switch Engine to GPU
print("Original Engine",Config.engine)
Config.engine = pass_engine
Config.engine = cuda_engine
print("Running Engine",Config.engine)

# Constants
MATRIX_SIZE = 100 # Size of square grid
BLOCK_DIMS = 8 # CUDA block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # CUDA grid dimensions
P_LOCAL = 0.50 # probability of local diffusion
P_NON_LOCAL = 0.50 # probability of non-local diffusion
N_ITERS = 5 # number of iterations
CODE = """
    __global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
    {{
        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of index
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure cell is within the grid dimensions
        if (x < grid_size && y < grid_size) {{

            unsigned int thread_id = y * grid_size + x;                     // thread index in array

            // edges will be ignored as starting points
            unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

            // only look at this cell if it's already a 1
            if (grid_a[thread_id] == 1) {{
                grid_b[thread_id] = 1;                                      // current cell
                if (!edge) {{
                    if (randoms[thread_id - grid_size] < prob) {{
                        grid_b[thread_id - grid_size] = 1;                  // above
                    }}
                    if (randoms[thread_id - grid_size - 1] < prob) {{
                        grid_b[thread_id - grid_size - 1] = 1;              // above and left
                    }}
                    if (randoms[thread_id - grid_size + 1] < prob) {{
                        grid_b[thread_id - grid_size + 1] = 1;              // above and right
                    }}
                    if (randoms[thread_id + grid_size] < prob) {{
                        grid_b[thread_id + grid_size] = 1;                  // below
                    }}
                    if (randoms[thread_id + grid_size - 1] < prob) {{
                        grid_b[thread_id + grid_size - 1] = 1;              // below and left
                    }}
                    if (randoms[thread_id + grid_size + 1] < prob) {{
                        grid_b[thread_id + grid_size + 1] = 1;              // below and right
                    }}
                    if (randoms[thread_id - 1] < prob) {{
                        grid_b[thread_id - 1] = 1;                          // left
                    }}
                    if (randoms[thread_id + 1] < prob) {{
                        grid_b[thread_id + 1] = 1;                          // right
                    }}
                }}
            }}
        }}
    }}

    __global__ void non_local_diffuse(float* grid_a, float* grid_b, float* randoms, int* x_coords, int* y_coords)
    {{
        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of index
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure cell is within the grid dimensions
        if (x < grid_size && y < grid_size) {{

            unsigned int thread_id = y * grid_size + x;                     // thread index in array

            // only look at this cell if its already a 1
            if (grid_a[thread_id] == 1) {{
                grid_b[thread_id] = 1;                                      // current cell
                if (randoms[thread_id] < prob) {{
                    unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
                    grid_b[spread_index] = 1;
                }}
            }}
        }}
    }}
"""

# Format code with constants and compile kernel
KERNEL_CODE = CODE.format(MATRIX_SIZE, P_LOCAL, MATRIX_SIZE, P_NON_LOCAL)
MOD = SourceModule(KERNEL_CODE)

# Get local and non-local diffusion functions from kernel
LOCAL = MOD.get_function('local_diffuse')
NON_LOCAL = MOD.get_function('non_local_diffuse')

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(empty_grid.size(MATRIX_SIZE) == initialize_grid.size(MATRIX_SIZE) < local_diffusion.vars(LOCAL, MATRIX_SIZE, GRID_DIMS, BLOCK_DIMS) == non_local_diffusion.vars(NON_LOCAL, MATRIX_SIZE, GRID_DIMS, BLOCK_DIMS) > AGStore.file("output.tif"))

