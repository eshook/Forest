"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from ..engines import Config
from ..bobs.Bob import *
import copy

class Primitive(object):

    def __init__(self, name = None):
        # Set name
        self.name = self.__class__.__name__
        if name is not None:
            self.name = name

    def __repr__(self):
        return self.name

    def __call__(self):
        print("__call__", self.name)
    
    def __eq__(self, right):
        print(self, "== (Sequence)", right)
        Config.engine.run(self)
        return right

    def __ne__(self, right):
        print(self, "!= (Synchronization)", right)
        Config.engine.run(self)
        Config.engine.synchronization()
        return right

    def __gt__(self, right):
        print(self, "> (Merge)", right)
        Config.engine.run(self)
        Config.engine.merge()
        return right

    def __lt__(self, right):
        print(self, "< (Split)", right)        
        Config.engine.run(self)
        Config.engine.split()
        return right

    # This wrap function makes the primitive callable with 2 parameters
    def wrap(self,l,r):
        Config.engine.stack.push(r)
        Config.engine.stack.push(l)
        self.__call__()
        return Config.engine.stack.pop()

# Pop 2 bobs off the data stack, apply function (func), then push 1 Bob back on the stack
def pop2(func):
    l = Config.engine.stack.pop()
    r = Config.engine.stack.pop()
    o = func(l,r)
    Config.engine.stack.push(o)

# Pop 2 bobs off the data stack, copy top bob for output, apply function (func) to data of each bob, 
# Save output to data of output bob, then push output bob back on the stack
def pop2data(func):
    l = Config.engine.stack.pop()
    r = Config.engine.stack.pop()
    o = copy.deepcopy(l)  # Make an output bob by copying the first bob on the stack (l)
    o.data = func(l.data,r.data)
    Config.engine.stack.push(o)

# This function exposes the engine's run to the outside world.
def run_primitive(op):
    return Config.engine.run(op)
    
