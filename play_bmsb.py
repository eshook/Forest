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
from pycuda.characterize import sizeof
import matplotlib.pyplot as plt
from os import path
import sys

# Switch Engine to GPU
Config.engine = cuda_engine
print("Running Engine",Config.engine)

# Files to use as initial population and survival probabilities
initial_population_file = '/home/iaa/bures024/Forest/2000_init_pop.tif'
survival_probabilities_file = '/home/iaa/bures024/Forest/2000_surv_probs.tif'

# Make sure files exist
if (path.isfile(initial_population_file) == False) or (path.isfile(survival_probabilities_file) == False):
    print('Error. File not found. Exiting program...')
    sys.exit()

# Load initial population and survival layer probabilities
initial_population = plt.imread(initial_population_file).astype(np.float32)
survival_probabilities = plt.imread(survival_probabilities_file).astype(np.float32)
survival_probabilities = np.divide(survival_probabilities,255)

# Make sure initial population and survival layer probabilities grids are square (n x n)
if (initial_population.shape[0] != initial_population.shape[1]) or (survival_probabilities.shape[0] != survival_probabilities.shape[1]):
    print('Invalid dimensions. Grid must be square (n x n dimensions). Exiting program...')
    sys.exit()

# Make sure initial population and survival layer probabilities grids are the same dimensions
if (initial_population.shape[0] != survival_probabilities.shape[0]) or (initial_population.shape[1] != survival_probabilities.shape[1]):
    print('Invalid entry. Initial population grid and survival probabilities grid must be same shape. Exiting program...')
    sys,exit()

# Constants
matrix_size = initial_population.shape[0]                   # Size of square grid
block_dims = 32                                             # CUDA block dimensions - maximum dimensions = 32 x 32
grid_dims = (matrix_size + block_dims - 1) // block_dims    # CUDA grid dimensions
p_local = 0.50                                              # probability an agent spreads during local diffusion
p_non_local = 0.33                                          # probability an agent spreads during non-local diffusion
growth_rate = 0.25                                          # expnential growth rate of population layer
mu = 0.0                                                    # location parameter of cauchy distribution
gamma = 1.0                                                 # scale parameter of cauchy distribution
n_iters = 1                                                 # number of iterations
kernel_code = """
    #include <curand_kernel.h>
    #include <math.h>

    extern "C" {

    __device__ float get_random_number(curandState* global_state, int thread_id) {

        curandState local_state = global_state[thread_id];
        float num = curand_uniform(&local_state);
        global_state[thread_id] = local_state;
        return num;
    }

    __device__ float get_random_angle_in_radians(curandState* global_state, int thread_id) {

        float radians = get_random_number(global_state, thread_id) * 2 * M_PI;
        return radians;
    }

    __device__ float get_random_cauchy_distance(curandState* global_state, int thread_id, float mu, float gamma) {

        float distance = fabsf(mu + gamma * tan(M_PI * (get_random_number(global_state,thread_id) - 0.5)));
        return distance;
    }

    __device__ int get_x_coord(int x, float radians, float distance) {

        int x_coord = (int) roundf(x + distance * sin(radians));
        return x_coord;
    }

    __device__ int get_y_coord(int y, float radians, float distance) {

        int y_coord = (int) roundf(y + distance * cos(radians));
        return y_coord;
    }

    __global__ void init_generators(curandState* global_state, int seed, int grid_size) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row index of cell

        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;
            curandState local_state;
            curand_init(seed, thread_id, 0, &local_state);
            global_state[thread_id] = local_state;
        }
    }

    __global__ void local_diffuse(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, int time) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of cell

        // make sure this cell is within bounds of grid
        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                     // thread index
            grid_b[thread_id] = grid_a[thread_id];                 // copy current cell
            int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

            // edges are ignored as starting points
            if (!edge) {

                // ignore cell if it is not already populated
                if (grid_a[thread_id] > 0.0) {

                    int count = 0;
                    int n_iters = grid_a[thread_id];
                    //__syncthreads();
                    float num;
                    int neighbor;

                    // each agent has a chance to spread
                    while (count < n_iters) {

                        num = get_random_number(global_state, thread_id);
                        
                        // this agent spreads to a neighbor
                        if (num < prob) {

                            // randomly select a neighbor
                            neighbor = (int) ceilf(get_random_number(global_state, thread_id) * 8.0);
                            
                            grid_b[thread_id] -= 1.0;
                            switch(neighbor) {
                                case 1:
                                    grid_b[thread_id - grid_size] += 1.0;           // above
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x, y - 1, time);
                                    break;
                                case 2:
                                    grid_b[thread_id - grid_size - 1] += 1.0;       // above and left
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x - 1, y - 1, time);
                                    break;
                                case 3:
                                    grid_b[thread_id - grid_size + 1] += 1.0;       // above and right
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x + 1, y - 1, time);
                                    break;
                                case 4:
                                    grid_b[thread_id + grid_size] += 1.0;           // below
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x, y + 1, time);
                                    break;
                                case 5:
                                    grid_b[thread_id + grid_size - 1] += 1.0;       // below and left
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x - 1, y + 1, time);
                                    break;
                                case 6:
                                    grid_b[thread_id + grid_size + 1] += 1.0;       // below and right
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x + 1, y + 1, time);
                                    break;
                                case 7:
                                    grid_b[thread_id - 1] += 1.0;                   // left
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x - 1, y, time);
                                    break;
                                case 8:
                                    grid_b[thread_id + 1] += 1.0;                   // right
                                    //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x + 1, y, time);
                                    break;
                                default:
                                    //printf("Invalid number encountered\\n");
                                    break;
                            }
                        }
                        count += 1;
                    }
                }
            }
        }
    }

    __global__ void non_local_diffuse(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, float mu, float gamma, int time) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row index of cell
 
        // make sure this cell is within bounds of grid
        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                     // thread index
            grid_b[thread_id] = grid_a[thread_id];                 // copy current cell

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0.0) {

                int count = 0;
                int n_iters = grid_a[thread_id];
                //__syncthreads();
                float num;
                float radians;
                float distance;
                int spread_index;
                int x_coord;
                int y_coord;

                // each agent has a chance to spread
                while (count < n_iters) {

                    num = get_random_number(global_state, thread_id);
                    
                    // this agent spreads to a neighbor
                    if (num < prob) {

                        // randomly select a cell
                        radians = get_random_angle_in_radians(global_state, thread_id);
                        distance = get_random_cauchy_distance(global_state, thread_id, mu, gamma);
                        x_coord = get_x_coord(x, radians, distance);
                        y_coord = get_y_coord(y, radians, distance);
                        //printf("Radians = %f\\tDistance = %f\\tX = %d\\tY = %d\\tX_coord = %d\\tY_coord = %d\\n", radians, distance, x, y, x_coord, y_coord);

                        // make sure chosen cell is in the grid dimensions and is not the current cell
                        if (x_coord < grid_size && x_coord >= 0 && y_coord < grid_size && y_coord >= 0 && (x_coord != x || y_coord != y)) {
                            spread_index = y_coord * grid_size + x_coord;
                            grid_b[thread_id] -= 1;
                            grid_b[spread_index] += 1;
                            //printf("Cell (%d,%d) spread to cell (%d,%d) at time %d\\n", x, y, x_coord, y_coord, time);
                        }
                    }
                    count += 1;
                }
            }
        }
    }

    __global__ void survival_of_the_fittest(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float* survival_probabilities, int time) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column index of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row index of cell

        // make sure this cell is within bounds of grid
        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                      // thread index
            grid_b[thread_id] = grid_a[thread_id];                  // copy current cell
            float num;

            // ignore cell if it is not already populated
            if (grid_a[thread_id] > 0.0) {

                num = get_random_number(global_state, thread_id);

                // agents in this cell die
                if (num < survival_probabilities[thread_id]) {
                    grid_b[thread_id] = 0.0;                        // cell dies
                    //printf("Cell (%d,%d) died at time %d (probability of death was %f)\\n", x, y, time, survival_probabilities[thread_id]);
                }
            }
        }
    }

    __global__ void population_growth(float* grid_a, float* grid_b, int grid_size, float growth_rate, int time) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;              // column index of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;              // row index of cell

        // make sure this cell is within bounds of grid
        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                      // thread index
            grid_b[thread_id] = grid_a[thread_id];                  // copy current cell
            //printf("Value at (%d,%d) is %f\\n", x, y, grid_b[thread_id]);

            // ignore cell if initial population was 0
            if (grid_a[thread_id] > 0.0) {

                // growth formula: x(t) = x(t-1) * (1 + growth_rate)^time
                int pop = grid_a[thread_id];
                int add_pop = (int) truncf(pop * pow((1 + growth_rate), time));
                grid_b[thread_id] += add_pop;
                //printf("Cell (%d,%d)'s population grew by %d at time %d\\n", x, y, add_pop, time);
            }
        }
    }
    
    } // end extern "C"
"""

mod = SourceModule(kernel_code, no_extern_c = True)

# Get kernel functions
local = mod.get_function('local_diffuse')
non_local = mod.get_function('non_local_diffuse')
survival_layer = mod.get_function('survival_of_the_fittest')
population_layer = mod.get_function('population_growth')
init_generators = mod.get_function('init_generators')

generator = curandom.XORWOWRandomNumberGenerator()
data_type_size = sizeof(generator.state_type, "#include <curand_kernel.h>")
generator._state = drv.mem_alloc((matrix_size * matrix_size) * data_type_size)
seed = 123456789
init_generators(generator.state, np.int32(seed), np.int32(matrix_size),
    grid = (grid_dims, grid_dims), block = (block_dims, block_dims, 1))

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(
    empty_grid.vars(matrix_size) == 
    initialize_grid.vars(matrix_size, initial_population, survival_probabilities, generator) ==
    bmsb_stop_condition.vars(n_iters) <= 
    local_diffusion.vars(local, matrix_size, p_local, grid_dims, block_dims) == 
    non_local_diffusion.vars(non_local, matrix_size, p_non_local, mu, gamma, grid_dims, block_dims) ==
    survival_function.vars(survival_layer, matrix_size, grid_dims, block_dims) ==
    population_growth.vars(population_layer, matrix_size, growth_rate, grid_dims, block_dims) ==
    bmsb_stop >= 
    AGStore.file("output.tif")
    )

