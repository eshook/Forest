"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Tyler Buresh, bures024@umn.edu)
@contributors: <Contribute and add your name here!>
"""

from ..core.Bob import *
from ..core.Engine import *
from ..core import Config
import copy
import math

import numpy as np

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule




# CUDA Engine
class CUDAEngine(Engine):

    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(CUDAEngine,self).__init__("CUDAEngine")
        self.bob_stack = Stack()
        self.queue = Queue()
        self.generator = None # psuedorandom number generator
        self.continue_cycle = False # looping variable
        self.n_iters = 0 # number of iterations to run
        self.iters = 0 # number of iterations already run
        self.survival_probabilities = None
        
    def split(self):

        # Move survival_probabilities to GPU memory
        if self.survival_probabilities is not None:
            self.survival_probabilities = gpuarray.to_gpu(self.survival_probabilities)

        # Pop everything off the stack and move from CPU to GPU memory
        # Self.bob_stack contains bobs. Self.stack contains data
        temp_stack = Stack()
        while self.stack.notempty():
            bob = self.stack.pop()
            self.bob_stack.push(bob)
            gpu_bob = gpuarray.to_gpu(bob())
            temp_stack.push(gpu_bob)

        # Push data back onto stack to maintain order
        while temp_stack.notempty():
            gpu_bob = temp_stack.pop()
            self.stack.push(gpu_bob)
        
        # Data is split so set split to be true
        self.is_split = True
        
    # Merge (>)
    def merge(self):

        # Move survival_probabilities to CPU memory
        if self.survival_probabilities is not None:
            self.survival_probabilities = self.survival_probabilities.get()

        # Do the same thing as split, but in reverse. 
        # Pop everything off the stack and move from GPU to CPU memory
        temp_stack = Stack()
        while self.stack.notempty():
            gpu_bob = self.stack.pop()
            bob = gpu_bob.get()
            temp_stack.push(bob)

        # Push data back onto stack to maintain order
        while temp_stack.notempty():
            bob = temp_stack.pop()
            self.stack.push(bob)

        # Now that everything is merged set split to be false
        self.is_split = False

    # Sequence (==)
    def sequence(self):
        pass

    # Cycle Start (<=)
    def cycle_start(self):
        # Move from CPU to GPU memory and indicate we will be looping
        self.split()
        self.continue_cycle = True

    # Cycle Stop (>=)
    def cycle_termination(self):
        # Loop until desired number of iterations is reached
        while self.continue_cycle == True:
            copy_queue = Queue()
            # Pop primitives and run each again
            while self.queue.notempty():
                prim = self.queue.dequeue()
                copy_queue.enqueue(prim)
                self.run(prim)
            self.queue = copy_queue

        # Now that were done looping, call merge to move from GPU to CPU memory
        self.merge()


    # This method will run a single primitive operation
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Save a copy of each primitive the first time it runs so we can loop
        if self.continue_cycle == True and self.iters == 0:
            self.queue.enqueue(primitive)

        # Check is_split, if running split then loop over split stacks
        if self.is_split:
            # Right now just run the primitive no matter what
            primitive()
        else:
            # otherwise just run the primitive
            primitive()

cuda_engine = CUDAEngine()

if __name__ == '__main__':
    pass
