"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from ..engines import Config
from ..bobs.Bob import *

class Primitive(object):

    def __init__(self, name):
        
        # Set name
        self.name = name
        
        # FIXME: Need to think this through more
        # If this primitive should pass through Bobs, then enable it.
        # In most cases this should be false (i.e., don't pass through data)
        self.passthrough = False

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
        Config.engine.synchronization(Config.inputs)
        return right

    def __gt__(self, right):
        print(self, "> (Merge)", right)
        Config.engine.run(self)
        Config.engine.merge(Config.inputs)
        return right

    def __lt__(self, right):
        print(self, "< (Split)", right)        
        Config.engine.run(self)
        Config.engine.split(Config.inputs)
        return right

# This function exposes the engine's run to the outside world.
def run_primitive(op):
    return Config.engine.run(op)
    