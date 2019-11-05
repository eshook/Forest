"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from ..core.Primitive import *
from ..bobs.Bobs import *

import math
import numpy as np

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule


## Brown Marmorated Stink Bug Related Raster Primitives

# initialize_grid
class Initialize_grid(Primitive):
    def __call__(self):
         if not self.size:
             self.size = 4 # By default you get a grid of 4x4

         # Create grid and set initial population, survival probabilities, and random number generator
         grid = Raster(h=self.size,w=self.size,nrows=self.size,ncols=self.size)
         grid.data = self.initial_population
         Config.engine.survival_probabilities = self.survival_probabilities
         Config.engine.generator = self.generator

         #return grid 
         Config.engine.stack.push(grid)

    # Save size of the grid, initial population, survival layer probabilities, and random number generator
    def vars(self,size,init_pop,surv_probs,generator):
         self.size = size
         self.initial_population = init_pop
         self.survival_probabilities = surv_probs
         self.generator = generator
         return self # Must still return self so there is something to call

initialize_grid = Initialize_grid()

# empty_grid
class Empty_grid(Primitive):
    def __call__(self):
         if not self.size:
             self.size = 4 # By default you get a grid of 4x4

         # Create grid and convert data to np.float32 (necessary for GPU computation)
         grid = Raster(h=self.size,w=self.size,nrows=self.size,ncols=self.size)
         grid.data = grid.data.astype(np.float32)

         #return grid 
         Config.engine.stack.push(grid)

    # Save size of the grid parameter
    def vars(self,size):
         self.size = size
         return self # Must still return self so there is something to call

empty_grid = Empty_grid()

# local_diffusion
class Local_diffusion(Primitive):
    def __call__(self):

        time = np.int32(Config.engine.iters + 1)

        # This decorator will wrap the pop2data2gpu function around diff
        # It will pop off data (GPU'ified arrays), apply diff, and push data back onto stack
        @pop2data2gpu
        def diff(gpu_grid_a,gpu_grid_b):
            self.action(gpu_grid_a, gpu_grid_b, Config.engine.generator.state, self.size, self.prob, time,
                grid = (self.grid_dims, self.grid_dims), block = (self.block_dims, self.block_dims, 1))
            gpu_grid_a, gpu_grid_b = gpu_grid_b, gpu_grid_a

            return gpu_grid_a,gpu_grid_b

    # Save kernel function, matrix size, diffusion probability, grid size, and block size
    def vars(self,func,size,prob,grid,block):
        self.action = func
        self.size = np.int32(size)
        self.prob = np.float32(prob)
        self.grid_dims = grid
        self.block_dims = block
        return self # Must still return self so there is something to call

local_diffusion = Local_diffusion()

# distance_diffusion
class Non_local_diffusion(Primitive):
    def __call__(self):

        time = np.int32(Config.engine.iters + 1)

        # This decorator will wrap the pop2data2gpu function around diff
        # It will pop off data (GPU'ified arrays), apply diff, and push data back onto stack
        @pop2data2gpu
        def diff(gpu_grid_a,gpu_grid_b):
            self.action(gpu_grid_a, gpu_grid_b, Config.engine.generator.state, self.size, self.prob, self.mu, self.gamma, time,
                grid = (self.grid_dims, self.grid_dims), block = (self.block_dims, self.block_dims, 1))
            gpu_grid_a, gpu_grid_b = gpu_grid_b, gpu_grid_a

            return gpu_grid_a,gpu_grid_b 

    # Save kernel function, matrix size, diffusion probability, mu, gamma, grid size, and block size
    def vars(self,func,size,prob,mu,gamma,grid,block):
        self.action = func
        self.size = np.int32(size)
        self.prob = np.float32(prob)
        self.mu = np.float32(mu)
        self.gamma = np.float32(gamma)
        self.grid_dims = grid
        self.block_dims = block
        return self # Must still return self so there is something to call

non_local_diffusion = Non_local_diffusion()

# survival_function
class Survival_of_the_fittest(Primitive):
    def __call__(self):

        time = np.int32(Config.engine.iters + 1)

        # This decorator will wrap the pop2data2gpu function around diff
        # It will pop off data (GPU'ified arrays), apply diff, and push data back onto stack
        @pop2data2gpu
        def diff(gpu_grid_a,gpu_grid_b):
            self.action(gpu_grid_a, gpu_grid_b, Config.engine.generator.state, self.size, Config.engine.survival_probabilities, time,
                grid = (self.grid_dims, self.grid_dims), block = (self.block_dims, self.block_dims, 1))
            gpu_grid_a, gpu_grid_b = gpu_grid_b, gpu_grid_a

            return gpu_grid_a,gpu_grid_b

    # Save kernel function, matrix size, grid size, and block size
    def vars(self,func,size,grid,block):
        self.action = func
        self.size = np.int32(size)
        self.grid_dims = grid
        self.block_dims = block
        return self # Must still return self so there is something to call

survival_function = Survival_of_the_fittest()

# population_growth
class Population_growth(Primitive):
    def __call__(self):

        time = np.int32(Config.engine.iters + 1)

        # This decorator will wrap the pop2data2gpu function around diff
        # It will pop off data (GPU'ified array), appliy diff, and push data back onto stack
        @pop2data2gpu
        def diff(gpu_grid_a,gpu_grid_b):
            self.action(gpu_grid_a, gpu_grid_b, self.size, self.rate, time,
                grid = (self.grid_dims, self.grid_dims), block = (self.block_dims, self.block_dims, 1))
            gpu_grid_a, gpu_grid_b = gpu_grid_b, gpu_grid_a

            return gpu_grid_a,gpu_grid_b

    # Save kernel function, matrix size, growth rate, grid size, and block size
    def vars(self,func,size,rate,grid,block):
        self.action = func
        self.size = np.int32(size)
        self.rate = np.float32(rate)
        self.grid_dims = grid
        self.block_dims = block
        return self # Must still return self so there is something to call

population_growth = Population_growth()

# bmsb_stop_condition
class Bmsb_stop_condition(Primitive):
    def __call__(self):
        # set number of iterations to run
        Config.engine.n_iters = self.n_iters

    def vars(self, n):
        self.n_iters = n
        return self # Must still return self so there is something to call

bmsb_stop_condition = Bmsb_stop_condition()

# bmsb_stop
class Bmsb_stop(Primitive):
    def __call__(self):
        Config.engine.iters += 1
        # check if we want to run more iterations
        if Config.engine.iters >= Config.engine.n_iters:
            Config.engine.continue_cycle = False

bmsb_stop = Bmsb_stop()




