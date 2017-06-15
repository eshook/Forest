"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Just Bob
# (Or for those less cool folks a Bounding OBject :-)
class Bob(object):
    def __init__(self, y = 0, x = 0, h = 0, w = 0, s = 0, d = 0):
        self.y = y # y-axis (origin)
        self.x = x # x-axis (origin)           ___
        self.h = h # height (y-axis)        Y |\__\
        self.w = w # width  (x-axis)        X \|__|
        self.s = s # start time                 T
        self.d = d # duration (time)

        self.data = None # By default Bobs don't have any data

    def __repr__(self):
        return "Bob (%f,%f) [%f,%f]" % (self.y,self.x,self.h,self.w)

    def __call__(self):
        return self.data

