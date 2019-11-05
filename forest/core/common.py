"""
Copyright (c) 2019 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Before importing ForEST modules, we need to create a conditional import function
# It should register each capability in the Config.packages dictionary

import importlib

from . import Config

def forest_package_register(package,package_name):
    Config.package[package_name] = True

def conditional_import(module_string,package,package_name):
    try:
        module = __import__("module_{}".format(module_string))

        forest_package_register(package,package_name)
        
    except ImportError:
        print(" [ ERROR ] Could not load module:",module_string," so disabling package",package,package_name)


        
        
# This is a generic 'stack' data structure.
class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self,entity):
        print("PUSH",entity)
        return self.stack.append(entity)
        
    def pop(self):
        print("POP")
        return self.stack.pop()

    def notempty(self):
        return len(self.stack)>0

    def size(self):
        return len(self.stack)

    def __repr__(self):
        return "Stack("+str(len(self.stack))+")"

# This is a generic 'queue' data structure.
class Queue(object):
    def __init__(self):
        self.queue = []

    def enqueue(self,entity):
        print("ENQUEUE",entity)
        return self.queue.insert(0,entity)

    def dequeue(self):
        print("DEQUEUE")
        return self.queue.pop()

    def notempty(self):
        return len(self.queue)>0

    def size(self):
        return len(self.queue)

    def __repr__(self):
        return "Queue("+str(len(self.queue))+")"

if __name__ == '__main__':
    pass
