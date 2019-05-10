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
MATRIX_SIZE = 8 # Size of square grid
BLOCK_DIMS = 2 # CUDA block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # CUDA grid dimensions
P_LOCAL = 0.25 # probability of local diffusion
P_NON_LOCAL = 0.25 # probability of non-local diffusion
N_ITERS = 3 # number of iterations
CODE = """
    #include <curand_kernel.h>
    #include <math.h>

    extern "C" {{

    __global__ void local_diffuse(float* grid_a, float* grid_b, curandState* global_state)
    {{

        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of index
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            unsigned int thread_id = y * grid_size + x;                     // thread index
            float num;
            unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0) {{

                grid_b[thread_id] = grid_a[thread_id];                      // current cell

                // edges are ignored as starting points
                if (!edge) {{

                    curandState local_state = global_state[thread_id];      // state of thread's generator
                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size] += 1;                  // above
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size - 1] += 1;              // above and left
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size + 1] += 1;              // above and right
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size] += 1;                  // below
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size - 1] += 1;              // below and left
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size + 1] += 1;              // below and right
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id - 1] += 1;                          // left
                    }}

                    num = curand_uniform(&local_state);
                    if (num < prob) {{
                        grid_b[thread_id + 1] += 1;                          // right
                    }}

                    global_state[thread_id] = local_state;                  // save new generator state
                }}
            }}
        }}
    }}

    __global__ void non_local_diffuse(float* grid_a, float* grid_b, curandState* global_state) {{

        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of element
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            unsigned int thread_id = y * grid_size + x;                     // thread index
            float num;
            unsigned int x_coord;
            unsigned int y_coord;
            unsigned int spread_index;

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0) {{

                grid_b[thread_id] = grid_a[thread_id];                      // current cell

                curandState local_state = global_state[thread_id];          // state of thread's generator
                num = curand_uniform(&local_state);

                // non-local diffusion occurs until a num > prob is randomly generated
                while (num < prob) {{

                    // scale to grid dimensions
                    x_coord = (int) truncf(curand_uniform(&local_state) * (grid_size - 0.000001));
                    y_coord = (int) truncf(curand_uniform(&local_state) * (grid_size - 0.000001));
                    spread_index = y_coord * grid_size + x_coord;

                    // printf("Thread_ID  = %u\\tNum = %f\\tY_coord = %u\\tX_coord = %u\\n", thread_id, num, y_coord, x_coord);
                    grid_b[spread_index] += 1;
                    num = curand_uniform(&local_state);
                }}
                global_state[thread_id] = local_state;                      // save new generator state
            }}
        }}
    }}
    }}
"""

# Format code with constants and compile kernel
KERNEL_CODE = CODE.format(MATRIX_SIZE, P_LOCAL, MATRIX_SIZE, P_NON_LOCAL)
MOD = SourceModule(KERNEL_CODE, no_extern_c = True)

# Get local and non-local diffusion functions from kernel
LOCAL = MOD.get_function('local_diffuse')
NON_LOCAL = MOD.get_function('non_local_diffuse')

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(
    empty_grid.size(MATRIX_SIZE) == 
    initialize_grid.size(MATRIX_SIZE) ==
    bmsb_stop_condition.vars(N_ITERS) <= 
    local_diffusion.vars(LOCAL, MATRIX_SIZE, GRID_DIMS, BLOCK_DIMS) == 
    non_local_diffusion.vars(NON_LOCAL, MATRIX_SIZE, GRID_DIMS, BLOCK_DIMS) ==
    bmsb_stop >= 
    AGStore.file("output.tif")
    )


