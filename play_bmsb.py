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
Config.engine = cuda_engine
print("Running Engine",Config.engine)

# Constants
MATRIX_SIZE = 10 # Size of square grid
BLOCK_DIMS = 2 # CUDA block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # CUDA grid dimensions
P_LOCAL = 0.25 # probability of local diffusion
P_NON_LOCAL = 0.25 # probability of non-local diffusion
P_DEATH = 0.25 # probablity a cell dies after diffusion functions
GROWTH_RATE = 0.25 # expnential growth rate
N_ITERS = 5 # number of iterations
CODE = """
    #include <curand_kernel.h>
    #include <math.h>

    extern "C" {{

    __device__ float get_random_number(curandState* global_state, int thread_id) {{
        
        curandState local_state = global_state[thread_id];
        float num = curand_uniform(&local_state);
        global_state[thread_id] = local_state;

        //printf("Random number = %f\\n", num);

        return num;

    }}

    __device__ float get_random_angle_in_radians(curandState* global_state, int thread_id) {{

        float radians = get_random_number(global_state, thread_id) * 2 * M_PI;
        return radians;
    }}

    __device__ float get_random_cauchy_distance(curandState* global_state, int thread_id, int mu, int gamma) {{

        float distance = fabsf(mu + gamma * tan(M_PI * (get_random_number(global_state,thread_id) - 0.5)));
        return distance;
    }}

    __device__ int get_x_coord(int x, float radians, float distance) {{

        int x_coord = (int) roundf(x + distance * sin(radians));
        return x_coord;
    }}

    __device__ int get_y_coord(int y, float radians, float distance) {{

        int y_coord = (int) roundf(y + distance * cos(radians));
        return y_coord;
    }}

    __global__ void local_diffuse(float* grid_a, float* grid_b, curandState* global_state)
    {{

        int grid_size = {};
        float prob = {};

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of index
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            int thread_id = y * grid_size + x;                     // thread index
            float num;
            int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

            grid_b[thread_id] = grid_a[thread_id];                      // current cell

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0.0) {{

                // edges are ignored as starting points
                if (!edge) {{

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size] += 1.0;                 // above
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size - 1] += 1.0;             // above and left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - grid_size + 1] += 1.0;             // above and right
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size] += 1.0;                 // below
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size - 1] += 1.0;             // below and left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + grid_size + 1] += 1.0;             // below and right
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id - 1] += 1.0;                         // left
                    }}

                    num = get_random_number(global_state, thread_id);
                    if (num < prob) {{
                        grid_b[thread_id + 1] += 1.0;                         // right
                    }}
                }}
            }}
        }}
    }}

    __global__ void non_local_diffuse(float* grid_a, float* grid_b, curandState* global_state) {{

        int grid_size = {};
        float prob = {};

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of element
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            int thread_id = y * grid_size + x;                     // thread index
            float num;
            float radians;
            float distance;
            int spread_index;
            int mu = 0;             //****** Change later ******
            int gamma = 1;          //****** Change later ******
            int x_coord;
            int y_coord;

            grid_b[thread_id] = grid_a[thread_id];                      // current cell

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0.0) {{

                num = get_random_number(global_state, thread_id);
                
                // non-local diffusion occurs if a num > prob is randomly generated
                if (num < prob) {{

                    radians = get_random_angle_in_radians(global_state, thread_id);
                    distance = get_random_cauchy_distance(global_state, thread_id, mu, gamma);
                    x_coord = get_x_coord(x, radians, distance);
                    y_coord = get_y_coord(y, radians, distance);
                    printf("Radians = %f\\tDistance = %f\\tX = %d\\tY = %d\\tX_coord = %d\\tY_coord = %d\\n", radians, distance, x, y, x_coord, y_coord);

                    if (x_coord < grid_size && x_coord >= 0 && y_coord < grid_size && y_coord >= 0 && (x_coord != x || y_coord != y)) {{
                        spread_index = y_coord * grid_size + x_coord;
                        grid_b[spread_index] += 1;
                        printf("Cell (%d,%d) spread to cell (%d,%d)\\n", x, y, x_coord, y_coord);
                    }}
                }}
            }}
        }}
    }}

    __global__ void survival_of_the_fittest(float* grid_a, float* grid_b, curandState* global_state) {{

        int grid_size = {};
        float prob = {};

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of element
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of index

        // make sure the current thread is within bounds of grid
        if (x < grid_size && y < grid_size) {{

            int thread_id = y * grid_size + x;
            float num;

            grid_b[thread_id] = grid_a[thread_id];

            if (grid_a[thread_id] > 0.0) {{

                num = get_random_number(global_state, thread_id);

                if (num < prob) {{
                    grid_b[thread_id] = 0.0;                        // cell dies
                }}
            }}
        }}
    }}

    __global__ void population_growth(float* initial_population, float* grid_a, float* grid_b, int* time) {{

        int grid_size = {};
        float growth_rate = {};

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < grid_size && y < grid_size) {{

            int thread_id = y * grid_size + x;

            if (initial_population[thread_id] > 0.0) {{

                //xt = x0(1 + r)^t
                int x0 = initial_population[thread_id];
                int xt = (int) truncf(x0 * pow((1 + growth_rate), time[0]));
                grid_b[thread_id] = grid_a[thread_id] + xt;
                //printf("Initial population of cell %d = %f\\tPopulation increased by %d\\tNew population = %f\\n", thread_id, initial_population[thread_id], xt, grid_b[thread_id]);
            }}
        }}
    }}
    }}
"""

# Format code with constants and compile kernel
KERNEL_CODE = CODE.format(MATRIX_SIZE, P_LOCAL, MATRIX_SIZE, P_NON_LOCAL, MATRIX_SIZE, P_DEATH, MATRIX_SIZE, GROWTH_RATE)
MOD = SourceModule(KERNEL_CODE, no_extern_c = True)

# Get kernel functions
LOCAL = MOD.get_function('local_diffuse')
NON_LOCAL = MOD.get_function('non_local_diffuse')
SURVIVAL = MOD.get_function('survival_of_the_fittest')
POPULATION_GROWTH = MOD.get_function('population_growth')

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(
    empty_grid.size(MATRIX_SIZE) == 
    initialize_grid.size(MATRIX_SIZE) ==
    bmsb_stop_condition.vars(N_ITERS) <= 
    #local_diffusion.vars(LOCAL, GRID_DIMS, BLOCK_DIMS) == 
    non_local_diffusion.vars(NON_LOCAL, GRID_DIMS, BLOCK_DIMS) ==
    #survival_function.vars(SURVIVAL, GRID_DIMS, BLOCK_DIMS) ==
    #population_growth.vars(POPULATION_GROWTH, GRID_DIMS, BLOCK_DIMS) ==
    bmsb_stop >= 
    AGStore.file("output.tif")
    )

'''
Possible kernel_types are: random distribution, cauchy distribution
Kernel params for random distribution: none
Kernel params for cauchy distribution: location, scale
local_diffusion.vars(LOCAL, GRID_DIMS, BLOCK_DIMS, kernel_type, kernel_params) == 
'''

'''
from PIL import Image
im = Image.open('example.tif')
imarray = numpy.array(im)
print('Shape = ', imarray.shape)
print('Size = ', imarray.size)
'''

'''
import gdal
example = gdal.Open('example.tif')
example_array = np.array(example.getRasterBand(1).ReadAsArray())
print('Shape = ', example_array.shape)
print('Size = ', example_array.size)
'''
'''
import rasterio
with rasterio.open('example.tif', 'r') as ds:
    arr = ds.read()
print('Shape = ', arr.shape)
print('Arr = ', arr)
'''

#population_grid = gdal.Open('eample.tif')
#POPULATION_GRID_ARRAY = np.array(population_grid.getRasterBand(1).ReadAsArray())

#survival_function.vars(SURVIVAL, GRID_DIMS, BLOCK_DIMS, SURVIVAL_LAYER_ARRAY)
#initialize_grid.size(MATRIX_SIZE, POPULATION_GRID_ARRAY)

'''
select a direction at random with equal likelihood
float angle = [random float between 0 and 1] * 2 * M_PI

choose distance to diffuse one agent from cauchy distribution
dist = mu + gamma * tan(M_PI([random float between 0 and 1] - 0.5))

subtract one agent from source cell, add one agent to target cell
'''


