"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# TODO: Replace individual variables with numpy arrays
#       use accessors to get access to individual values
#       Need to profile to see if this improves overall speedup.
#       This would align with other projects that use arrays to process x,y,z using vectorization
#       In particular, I think this could be helpful for using Bobs for indexing (e.g., r-tree, etc.)

# Just Bob
# (Or for those less cool folks a Spatial-Temporal Bounding OBject :-)
class Bob(object):
    def __init__(self, y = 0, x = 0, h = 0, w = 0, t = 0, d = 0):
        self.y = y # y-axis (origin)
        self.x = x # x-axis (origin)           ___
        self.h = h # height (y-axis)        Y |\__\
        self.w = w # width  (x-axis)        X \|__|
        self.t = t # t-axis (origin)             T
        self.d = d # duration (t-axis)

        self.createdby = "" # Who or what created these data

        self.data = None # By default Bobs don't have any data


    def __repr__(self):
        return "Bob (%f,%f) [%f,%f]" % (self.y,self.x,self.h,self.w)

    def __call__(self):
        return self.data
