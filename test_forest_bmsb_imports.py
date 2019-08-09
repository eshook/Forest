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
matrix_size = 10 # Size of square grid
block_dims = 4 # CUDA block dimensions
grid_dims = (matrix_size + block_dims - 1) // block_dims # CUDA grid dimensions
p_local_always = 1.0 # probability of local diffusion
p_local_never = 0.0
p_non_local_always = 1.0 # probability of non-local diffusion
p_non_local_never = 0.0
p_death_none = 1.0 # probablity a cell dies after diffusion functions
p_death_all = 0.0
growth_rate = 0.5
n_iters = 5 # number of iterations
mu = 0.0
gamma = 1.0
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

    __global__ void test_get_random_number(curandState* global_state, float* grid, int grid_size) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of cell

        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                     // thread index
            grid[thread_id] = get_random_number(global_state, thread_id);
        }
    }

    __device__ float get_random_angle_in_radians(curandState* global_state, int thread_id) {

        float radians = get_random_number(global_state, thread_id) * 2 * M_PI;
        return radians;
    }

    __global__ void test_get_random_angle_in_radians(curandState* global_state, float* grid, int grid_size) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;             // column element of cell
        int y = threadIdx.y + blockIdx.y * blockDim.y;             // row element of cell

        if (x < grid_size && y < grid_size) {

            int thread_id = y * grid_size + x;                     // thread index
            grid[thread_id] = get_random_angle_in_radians(global_state, thread_id);
        }
    }

    __device__ float get_random_cauchy_distance(curandState* global_state, int thread_id, float mu, float gamma) {

        float distance = fabsf(mu + gamma * tan(M_PI * (get_random_number(global_state,thread_id) - 0.5)));
        return distance;
    }

    __global__ void test_get_random_cauchy_distance(curandState* global_state, float* grid) {

        printf("\\n");
    }

    __device__ int get_x_coord(int x, float radians, float distance) {

        int x_coord = (int) roundf(x + distance * sin(radians));
        return x_coord;
    }

    __global__ void test_get_x_coord(float* grid) {

        printf("\\n");
    }

    __device__ int get_y_coord(int y, float radians, float distance) {

        int y_coord = (int) roundf(y + distance * cos(radians));
        return y_coord;
    }

    __global__ void test_get_y_coord(float* grid) {

        printf("\\n");
    }

    __global__ void local_diffuse_always(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, int time) {

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

                            // hard code a given neighbor for unit testing purposes
                            neighbor = 4;
                            
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

    __global__ void local_diffuse_never(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, int time) {

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

    __global__ void non_local_diffuse_always(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, float mu, float gamma, int time) {

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

                        // hard code a cell for unit testing purposes
                        x_coord = 0;
                        y_coord = 0;

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

    __global__ void non_local_diffuse_never(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float prob, float mu, float gamma, int time) {

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

    __global__ void survival_of_the_fittest_none_survive(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float* survival_probabilities, int time) {

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

    __global__ void survival_of_the_fittest_all_survive(float* grid_a, float* grid_b, curandState* global_state, int grid_size, float* survival_probabilities, int time) {

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
local_always = mod.get_function('local_diffuse_always')
local_never = mod.get_function('local_diffuse_never')
non_local_always = mod.get_function('non_local_diffuse_always')
non_local_never = mod.get_function('non_local_diffuse_never')
survival_none = mod.get_function('survival_of_the_fittest_none_survive')
survival_all = mod.get_function('survival_of_the_fittest_all_survive')
population_growth = mod.get_function('population_growth')
get_random_number = mod.get_function('test_get_random_number')
get_random_angle = mod.get_function('test_get_random_angle_in_radians')
get_random_distance = mod.get_function('test_get_random_cauchy_distance')
get_x = mod.get_function('test_get_x_coord')
get_y = mod.get_function('test_get_y_coord')

