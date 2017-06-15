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
        run_primitive(self)
        return right

    def __gt__(self, right):
        print(self, "> (Merge)", right)
        run_primitive(self)
        return right

    def __lt__(self, right):
        print(self, "< (Split)", right)
        run_primitive(self)
        return right

        
# This method will run a single primitive operation
# It will pull data from inputs and run the primitive
# It will save the input
def run_primitive(op):
    #global inputs  # Grab the global inputs variable
    inputs = Config.inputs

    # Uncomment for debug statements
    #print("run_primitive inputs", inputs)
    #print("op(inputs)", op,"(",inputs,")")
    
    # Get the name of the primitive operation being executed
    name = op.__class__.__name__

    # Save the flows information in the global config data structure
    # FIXME: The problem with this solution is all data will be stored
    #        indefinitely, which is going to be a huge problem.
    Config.flows[name] = {}
    Config.flows[name]['input'] = inputs
    
    if isinstance(inputs,Bob): # If it is a bob
        inputs = op(inputs)    # Just pass in the bob
    else:                      # If it is a list
        #print("inputs=", inputs)
        inputs = op(*inputs)   # De-reference the list and pass as parameters

    # Save the outputs from this primitive
    Config.flows[name]['output'] = inputs
    
    # Uncomment to see debug information with 'flows'
    #print("flows",Config.flows)
    
    # Save inputs from this primitive, for the next primtive/pattern
    if op.passthrough is False: # Typical case
        Config.inputs = inputs # Reset the inputs
    else:
        Config.inputs.append(inputs) # Add to the inputs
        
    return inputs
