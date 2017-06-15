"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

class Engine(object):
    def __init__(self, engine_type):
        self.engine_type = engine_type # Describes the type of engine
        
    def __repr___(self):
        return "Engine "+str(self.engine_type)
        
    # Split (<)
    def split(self, bobs):
        pass
    
    # Merge (>)
    def merge(self, bobs):
        pass

    # Sequence (==)
    def sequence(self, bobs):
        pass
    
    # Synchronization (!=)
    def synchronization(self, bobs):
        pass
    
    # Cycle start (<<)
    def cycle_start(self, bobs):
        pass
    
    # Cycle termination (>>)
    def cycle_termination(self, bobs):
        pass