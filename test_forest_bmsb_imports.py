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

# Constants
MATRIX_SIZE = 10 # Size of square grid
BLOCK_DIMS = 2 # CUDA block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # CUDA grid dimensions
P_LOCAL = 1.0 # probability of local diffusion
P_NON_LOCAL = 0.25 # probability of non-local diffusion
P_DEATH = 0.25 # probablity a cell dies after diffusion functions
N_ITERS = 5 # number of iterations
CODE = """
    #include <curand_kernel.h>
    #include <math.h>

    extern "C" {{

    __device__ float get_random_number(curandState* global_state, unsigned int thread_id) {{
        
        curandState local_state = global_state[thread_id];
        float num = curand_uniform(&local_state);
        global_state[thread_id] = local_state;

        //printf("Thread = %u\\tRandom number = %f\\n", thread_id, num);

        return num;

    }}

    __device__ float get_random_cell(curandState* global_state, unsigned int thread_id, unsigned int grid_size) {{
        
        unsigned int x = (int) truncf(get_random_number(global_state, thread_id) * (grid_size - 0.000001));
        unsigned int y = (int) truncf(get_random_number(global_state, thread_id) * (grid_size - 0.000001));
        
        //printf("Thread = %u\\tRandom x = %u\\tRandom y = %u\\n", thread_id, x, y);
        
        return y * grid_size + x;
    }}

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

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size] += 1;                 // above
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size - 1] += 1;             // above and left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size + 1] += 1;             // above and right
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size] += 1;                 // below
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size - 1] += 1;             // below and left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size + 1] += 1;             // below and right
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - 1] += 1;                         // left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + 1] += 1;                         // right
                    }}
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
            unsigned int spread_index;

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0) {{

                grid_b[thread_id] = grid_a[thread_id];                      // current cell

                num = get_random_number(global_state, thread_id);

                // non-local diffusion occurs until a num > prob is randomly generated
                while (num < prob) {{

                    spread_index = get_random_cell(global_state, thread_id, grid_size);
                    grid_b[spread_index] += 1;
                    num = get_random_number(global_state, thread_id);
                }}
            }}
        }}
    }}

    __global__ void survival_of_the_fittest(float* grid_a, float* grid_b, curandState* global_state) {{

        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of element
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            unsigned int thread_id = y * grid_size + x;
            float num;

            if (grid_a[thread_id] > 0) {{

                num = get_random_number(global_state, thread_id);

                if (num < prob) {{
                    grid_b[thread_id] = 0;                                  // cell dies
                }}
            }}
        }}
    }}
    }}
"""

# Format code with constants and compile kernel
KERNEL_CODE = CODE.format(MATRIX_SIZE, P_LOCAL, MATRIX_SIZE, P_NON_LOCAL, MATRIX_SIZE, P_DEATH)
MOD = SourceModule(KERNEL_CODE, no_extern_c = True)

# Get kernel functions
LOCAL = MOD.get_function('local_diffuse')
NON_LOCAL = MOD.get_function('non_local_diffuse')
SURVIVAL = MOD.get_function('survival_of_the_fittest')

