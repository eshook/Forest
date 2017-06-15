"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""


# This is a global variable that allows the outputs from one interop 
# to be passed as 'inputs' to the next interop
inputs = []

# This is a global variable that goes beyond inputs. 
# It is a register for data flows that can be 'tapped' if needed
flows = {}

# This is a global variable that enables users and developers
# to control which computational engine is used
# When set to none, the pattern operators (< > !=) do nothing.
engine = None
