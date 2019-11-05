"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Luyi Hunter, chen3461@umn.edu; Xinran Duan, duanx138@umn.edu)
@contributors: <Contribute and add your name here!>
"""

from .Bob import *
from .common import *
from . import Config
import copy

'''
import math
import multiprocessing

import numpy as np
import gdal

'''

class Engine(object):
    def __init__(self, engine_type):
        self.engine_type = engine_type # Describes the type of engine
        self.is_split = False # The data are not split at the beginning

        self.stack = Stack() # Data stack
        
    def __repr___(self):
        return "Engine "+str(self.engine_type)
        
    # Split (<)
    # Split has two possible consequences:
    # (1) Modify the data stack by splitting bobs and creating multiple data stacks
    # (2) Initiate parallelism, which can be applied to one or more of the split data stacks
    def split(self):
        pass
    
    # Merge (>)
    # Merge has two possible consequences:
    # (1) Modify the data stack by merging bobs from multiple data stacks and return to a single data stack
    # (2) End parallelism
    def merge(self):
        pass

    # Sequence (==)
    def sequence(self):
        pass
    
    # Synchronization (!=)
    def synchronization(self):
        pass
    
    # Cycle start (<<)
    def cycle_start(self):
        pass
    
    # Cycle termination (>>)
    def cycle_termination(self):
        pass

    # This method will run a single primitive operation
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Normally we would check is_split, but in the basic case we don't handle it.
        # So just call the primitive
        # The primitive will pop bobs off the stack and push outputs back on.
        primitive()


class PassEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(PassEngine,self).__init__("PassEngine")
    
# This is the default engine that doesn't do anything.
pass_engine = PassEngine()    

# By default use the pass_engine
Config.engine = pass_engine

if __name__ == '__main__':
    pass
