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

MATRIX_SIZE = 128
CODE = """
    __global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
    {{

        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;         // column element of index
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;         // row element of index
        unsigned int thread_id = y * grid_size + x;                     // thread index in array

        // edges will be ignored as starting points
        unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

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

    __global__ void non_local_diffuse(float* grid_a, float* grid_b, float* randoms, int* x_coords, int* y_coords)
    {{

        unsigned int grid_size = {};
        float prob = {};

        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;         // column element of index
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;         // row element of index
        unsigned int thread_id = y * grid_size + x;                     // thread index in array

        if (grid_a[thread_id] == 1) {{
            grid_b[thread_id] = 1;                                  // current cell
            if (randoms[thread_id] < prob) {{
                unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
                grid_b[spread_index] = 1;
            }}
        }}
    }}
"""

# Now run one iteration of the Brown Marmorated Stink Bug (BMSB) Diffusion Simulation
run_primitive(initialize_grid.size(MATRIX_SIZE) == empty_grid.size(MATRIX_SIZE) == initialize_kernel.kernel(CODE) < local_diffusion == non_local_diffusion > AGStore.file("output.tif"))


