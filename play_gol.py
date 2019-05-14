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
MATRIX_SIZE = 10 # Size of square grid
BLOCK_DIMS = 4 # CUDA block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # CUDA grid dimensions
N_ITERS = 5 # number of iterations
CODE = """
    // Update each cell of the grid
    // Any live cell with less than two live neighbors dies
    // Any live cell with two or three live neighbors lives
    // Any live cell with four or more live neighbors dies
    // Any dead cell with three neighbors becomes a live cell
    __global__ void life_step(float *board, float *board2)
    {{

        unsigned int m_size = {};
        unsigned int num_cells = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;                 // Column index of the element
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;                 // Row index of the element
        unsigned int thread_id = y * m_size + x;                                // Thread ID in the board array

        // Game of life classically takes place on an infinite grid
        // I've used a toroidal geometry for the problem
        // The matrix wraps from top to bottom and from left to right
        unsigned int above = (thread_id - m_size) % num_cells;
        unsigned int below = (thread_id + m_size) % num_cells;
        unsigned int left;
        if (thread_id % m_size == 0) {{
            left = thread_id + m_size - 1;
        }} else {{
            left = thread_id - 1;
        }}
        unsigned int right;
        if (thread_id % m_size == m_size - 1) {{
            right = thread_id - m_size + 1;
        }} else {{
            right = thread_id + 1;
        }}
        unsigned int above_left;
        if (thread_id % m_size == 0) {{
            above_left = (thread_id - 1) % num_cells;
        }} else {{
            above_left = (thread_id - m_size - 1) % num_cells;
        }}
        unsigned int above_right;
        if (thread_id % m_size == m_size - 1) {{
            above_right = (thread_id - blockDim.x * m_size + 1) % num_cells;
        }} else {{
            above_right = (thread_id - m_size + 1) % num_cells;
        }}
        unsigned int below_left;
        if (thread_id % m_size == 0) {{
            below_left = (thread_id + blockDim.x * m_size - 1) % num_cells;
        }} else {{
            below_left = (thread_id + m_size - 1) % num_cells;
        }}
        unsigned int below_right;
        if (thread_id % m_size == m_size - 1) {{
            below_right = (thread_id + 1) % num_cells;
        }} else {{
            below_right = (thread_id + m_size + 1) % num_cells;
        }}

        unsigned int num_neighbors = board[above] + board[below] + board[left] + board[right] +
            board[above_left] + board[above_right] + board[below_left] + board[below_right];

        unsigned int live_and2 = board[thread_id] && (num_neighbors == 2);          // Live cell with 2 neighbors
        unsigned int live_and3 = board[thread_id] && (num_neighbors == 3);          // Live cell with 3 neighbors
        unsigned int dead_and3 = !(board[thread_id]) && (num_neighbors == 3);       // Dead cell with 3 neighbors
        board2[thread_id] = live_and2 || live_and3 || dead_and3;
    }}
"""

# Format code with constants and compile kernel
KERNEL_CODE = CODE.format(MATRIX_SIZE, MATRIX_SIZE * MATRIX_SIZE)
MOD = SourceModule(KERNEL_CODE)

# Get kernel functions
STEP = MOD.get_function('life_step')

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(
    empty_grid.size(MATRIX_SIZE) == 
    game_of_life_grid.size(MATRIX_SIZE) ==
    bmsb_stop_condition.vars(N_ITERS) <= 
    game_of_life.vars(STEP, GRID_DIMS, BLOCK_DIMS) ==
    bmsb_stop >= 
    AGStore.file("output.tif")
    )


