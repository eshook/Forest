"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""
from .Primitive import *

class Pattern(Primitive):
    def __init__(self, name):
        # Call Primitive.__init__ as the super class
        super(Pattern,self).__init__(name)
        
        # Reset the name
        self.name = "Pattern "+name
