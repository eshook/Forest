"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# This is a global variable that enables users and developers
# to control which computational engine is used
# When set to pass_engine, the pattern operators (< > !=) do nothing.
engine = None

# Packages that are available.
packages = {'bobs':{}, 'engines':{}, 'patterns':{}, 'primitives':{}}


# Preferred number of tiles to split bobs into
n_tiles = 20

# Preferred number of cores to use in parallel computing
n_cores = 4
