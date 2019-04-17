"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Luyi Hunter, chen3461@umn.edu; Xinran Duan, duanx138@umn.edu)
@contributors: <Contribute and add your name here!>
"""

from ..bobs.Bob import *
from ..bobs.Bobs import *
from . import Config
import math
import multiprocessing

import numpy as np
import gdal

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

    def __repr__(self):
        return "Stack("+str(len(self.stack))+")"

class Engine(object):
    def __init__(self, engine_type):
        self.engine_type = engine_type # Describes the type of engine
        self.is_split = False # The data are not split at the beginning

        self.stack = Stack() # Data stack


        
    def __repr___(self):
        return "Engine "+str(self.engine_type)
        
    # Split (<)
    # Split has two possible consequences:
    # (1) Modify the data stack by splitting bobs and creating multiple data stacks
    # (2) Initiate parallelism, which can be applied to one or more of the split data stacks
    def split(self):
        pass
    
    # Merge (>)
    # Merge has two possible consequences:
    # (1) Modify the data stack by merging bobs from multiple data stacks and return to a single data stack
    # (2) End parallelism
    def merge(self):
        pass

    # Sequence (==)
    def sequence(self):
        pass
    
    # Synchronization (!=)
    def synchronization(self):
        pass
    
    # Cycle start (<<)
    def cycle_start(self):
        pass
    
    # Cycle termination (>>)
    def cycle_termination(self):
        pass

    # This method will run a single primitive operation
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Normally we would check is_split, but in the basic case we don't handle it.
        # So just call the primitive
        # The primitive will pop bobs off the stack and push outputs back on.
        primitive()


class PassEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(PassEngine,self).__init__("PassEngine")
    
# This is the default engine that doesn't do anything.
pass_engine = PassEngine()    


class TileEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(TileEngine,self).__init__("TileEngine")

        self.split_stacks = [] # List of Data stacks to maintain (only when split into tiles)
        
    # Split (<)
    def split(self):

        # If already split, do nothing.
        if self.is_split is True:
            return
        
        # Set split to True so engine knows that the data stack has been split 
        self.is_split = True 

        num_tiles = Config.n_cores # The the number of tiles to split into as the number of cores

        # Make a list of data stacks for the split
        self.split_stacks = []
        for i in range(num_tiles):
            self.split_stacks.append(Stack()) 

        print("split_stacks",self.split_stacks)


        # Here we need to be careful, because the order of the data stack is important.
        # To maintain order while splitting everything we will make a temporary stack
        # for bobs "to be split"

        tobesplit_stack = Stack()
        while self.stack.notempty():
            # Take a bob off the current stack and push it on the tobesplit stack
            bob = self.stack.pop()
            tobesplit_stack.push(bob)

        # Now the data stack is empty and we can begin splitting all the bobs
        while tobesplit_stack.notempty():
            bob = tobesplit_stack.pop() # Pop off a bob

            # FIXME: This needs to be handled better.
            if not isinstance(bob,Raster): # If it isn't a Raster, thne don't try to split it. Just copy it...
                for i in range(num_tiles):
                    self.split_stacks[i].push(bob) # Just push entire bob on each split stack. 
                    
                continue # Now skip to the next bob in the list

            # Now we are dealing with a Raster
            # Set tile nrows and ncols using row decomposition for now
            tile_nrows = math.ceil(bob.nrows / num_tiles)
            tile_ncols = bob.ncols
            
            for tile_index in range(num_tiles):
                # Calculate the r,c location
                tile_r = tile_nrows * tile_index
                tile_c = 0
                
                # For the last tile_index, see if we are "too tall"
                # Meaning that the tiles are larger than the bob itself
                #  split Bob size      > Actual bob size
                if tile_r + tile_nrows > bob.nrows:
                    # If so, then resize so it is correct to bob.nrows
                    tile_nrows = bob.nrows - tile_r
                
                # Set tile height and width
                tile_h = bob.cellsize * tile_nrows
                tile_w = bob.w

                # Calculate y,x
                tile_y = bob.y + tile_r * bob.cellsize
                tile_x = bob.x
                
                # Create the tile
                tile = Bob(tile_y,tile_x,tile_h,tile_w)
                tile.nrows = tile_nrows
                tile.ncols = tile_ncols
                tile.r =     tile_r
                tile.c =     tile_c

                # Copy some attributes from the bob
                tile.cellsize =    bob.cellsize
                tile.filename =    bob.filename 
                tile.nodatavalue = bob.nodatavalue
                
                # Split the data (depends on raster/vector)
                tile.data = bob.get_data(tile_r,tile_c,tile_nrows,tile_ncols)
                ######################################################

                # Push this tile onto the split stack at it's index position
                self.split_stacks[tile_index].push(tile)

        print("stack",self.stack)
        print("tobestack",tobesplit_stack)
        print("splitstack",self.split_stacks)

        return # Then quit for now so code works

        # # # ## #  # # # ## # # # # # ## # # # # ### ## # # # # ## # # #

        # Otherwise start the split process
        
        # Set the number of tiles to split to
        # FIXME: Eventually this should be determined or user-defined.
        num_tiles = Config.n_tile
        print("-> Number of tiles = ", num_tiles)
        
        new_inputs = []
        # Loop over bobs in inputs to split
        for bob in Config.inputs:
            
            # For each bob, create a new split tile (TileEngine :)
            tiles = []
            
            # Split only works for rasters for now
            # For all other data types (e.g., vectors) we just duplicate the data
            if not isinstance(bob,Raster): # Check if not a raster

                '''
                # Temporarily removing STPoints and Points from the code before more vetting
				#spatio temporal point decomposition by box based splitting
                if isinstance(bob,STPoint):
                    #this should already be calculated and should be a global parameter
                    distancebufferinmeters,timebufferinmilliseconds=0,0
                    xranges=np.linspace(bob.x, bob.x+bob.w, num=num_tiles+1,endpoint=True)
                    yranges=np.linspace(bob.y, bob.y+bob.h, num=num_tiles+1,endpoint=True)
                    tranges=np.linspace(bob.s, bob.s+bob.d, num=num_tiles+1,endpoint=True,dtype=np.int64)
                    boxwidth,boxheight,boxduration=xranges[1]-xranges[0],yranges[1]-yranges[0],tranges[1]-tranges[0]
                    #create the boxes
                    for i in xrange(len(xranges)-1):
                        for j in xrange(len(yranges)-1):
                            for k in xrange(len(tranges)-1):
                                #buffer will be used during data addition
                                box=STPoint(yranges[j],xranges[i], boxheight, boxwidth, tranges[k], boxduration)
                                box.data=[]
                                tiles.append(box)
                    #This should be done in parallel, distribute the bob array, and boxes to cores, and finally merge to get filled bobs
                    for d in bob.data:
                        for box in tiles:
                            #check if the point is with in the buffer ranges
                            if d['x']>=np.max((box.x-distancebufferinmeters),bob.x) and d['x']<np.min((box.x-distancebufferinmeters+box.w+(2*distancebufferinmeters)),(bob.x+bob.w)) and d['y']>=np.max((box.y-distancebufferinmeters),bob.y) and d['y']<np.min((box.y-distancebufferinmeters+box.h+(2*distancebufferinmeters)),(bob.y+bob.h)) and d['t']>=np.max((box.s-timebufferinmilliseconds),bob.s) and d['t']<np.min((box.s-timebufferinmilliseconds+box.d+(2*timebufferinmilliseconds)),(bob.s+bob.d)):
                                #Check if with in original box, if with in box add to data, else to halozone
                                if d['x']>=box.x and d['x']<box.x+box.w and d['y']>=box.y and d['y']<box.y+box.h and d['t']>=box.s and d['t']<box.s+box.d:
                                    box.data.append(d)
                                else:
                                    box.halo.append(d)
                                continue
                    new_inputs.append(tiles)
                    continue
                #spatial point decomposition by 2D splitting
                if isinstance(bob,Point):
                    #this should already be calculated and should be a global parameter
                    distancebufferinmeters=0
                    xranges=np.linspace(bob.x, bob.x+bob.w, num=num_tiles+1,endpoint=True)
                    yranges=np.linspace(bob.y, bob.y+bob.h, num=num_tiles+1,endpoint=True)
                    boxwidth,boxheight=xranges[1]-xranges[0],yranges[1]-yranges[0]
                    #create the split boxes
                    for i in xrange(len(xranges)-1):
                        for j in xrange(len(yranges)-1):
                            #buffer will be used during data addition
                            box=Point(yranges[j],xranges[i], boxheight, boxwidth, 0,0)
                            box.data=[]
                            tiles.append(box)
                    #This should be done in parallel, distribute the bob array, and boxes to cores, and finally merge to get filled bobs
                    for d in bob.data:
                        for box in tiles:
                            #check if the point is with in the buffer ranges
                            if d['x']>=np.max((box.x-distancebufferinmeters),bob.x) and d['x']<np.min((box.x-distancebufferinmeters+box.w+(2*distancebufferinmeters)),(bob.x+bob.w)) and d['y']>=np.max((box.y-distancebufferinmeters),bob.y) and d['y']<np.min((box.y-distancebufferinmeters+box.h+(2*distancebufferinmeters)),(bob.y+bob.h)):
                                #Check if with in original box, if with in box add to data, else to halozone
                                if d['x']>=box.x and d['x']<box.x+box.w and d['y']>=box.y and d['y']<box.y+box.h:
                                    box.data.append(d)
                                else:
                                    box.halo.append(d)
                                continue
                    new_inputs.append(tiles)
                    continue
                '''

		# If it isn't a raster, then just copy the bob for each tile.

                for tile_index in range(num_tiles):
                    tiles.append(bob) # Just copy the entire bob to a tile list
                    
                new_inputs.append(tiles) # Now add the tiles to new_inputs
                continue # Now skip to the next bob in the list

            # This code will only be reached for Raster data types
            assert(isinstance(bob,Raster))
            # Sanity check, if tiles are larger than data
            if num_tiles > bob.nrows:
                num_tiles = bob.nrows # Reset to be 1 row per tile
            
            # Set tile nrows and ncols
            tile_nrows = math.ceil(bob.nrows / num_tiles)
            tile_ncols = bob.ncols
            
            for tile_index in range(num_tiles):
                # Calculate the r,c location
                tile_r = tile_nrows * tile_index
                tile_c = 0
                
                # For the last tile_index, see if we are "too tall"
                # Meaning that the tiles are larger than the bob itself
                #  split Bob size      > Actual bob size
                if tile_r + tile_nrows > bob.nrows:
                    # If so, then resize so it is correct to bob.nrows
                    tile_nrows = bob.nrows - tile_r
                
                # Set tile height and width
                tile_h = bob.cellsize * tile_nrows
                tile_w = bob.w

                # Calculate y,x
                tile_y = bob.y + tile_r * bob.cellsize
                tile_x = bob.x
                
                # Create the tile
                tile = Bob(tile_y,tile_x,tile_h,tile_w)
                tile.nrows = tile_nrows
                tile.ncols = tile_ncols
                tile.r =     tile_r
                tile.c =     tile_c
                tile.cellsize = bob.cellsize
                
                ######################################################
                ## Copy filename from Raster Bob to each tile
                tile.filename = bob.filename 
                tile.nodatavalue = bob.nodatavalue
                
                # FIXME: Need a better method to copy these over.
                
                # Split the data (depends on raster/vector)
                tile.data = bob.get_data(tile_r,tile_c,tile_nrows,tile_ncols)
                ######################################################
                                
                # Save tiles
                tiles.append(tile)
            # Save list of tiles (split Bobs) to new inputs
            # Notice that they are not grouped as inputs
            # So they will need to be zipped
            new_inputs.append(tiles)
                    
        # Now we have new_inputs so rewrite Config.inputs with new list
        # Zip the list to create groups of split bobs
        # These groups will be input for the primitives
        zip_inputs = zip(*new_inputs)
        Config.inputs = list(zip_inputs) # Dereference zip object and create a list
        
        # Set split to True so engine knows that Config.inputs is split                
        self.is_split = True 
        
    # Merge (>)
    def merge(self):
        # FIXME: This needs to be handled.
        # For now just copy the first stack in split_stack, and remove split_stack
        self.stack = self.split_stacks[0]
        self.split_stacks = []

        # Now that everything is merged set split to be false
        self.is_split = False

    # Sequence (==)
    def sequence(self):
        # If the Bobs are split, then handle it
        # If they are not, then there is nothing to do
        if self.is_split is True:
            # FIXME: Need to handle this
            print("NEED TO LOOP OVER SPLIT BOBS")
            pass # Loop over all the split Bobs
        pass

    # This method will run a single primitive operation
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Check is_split, if running split then loop over split stacks
        if self.is_split:
            # Loop over each split stack
            for i in range(len(self.split_stacks)):
                # Assign it as the main stack
                self.stack = self.split_stacks[i]

                # Run the primitive, which will apply itself to this split stack
                # including pushing everything back onto the stack
                primitive()

                # So once it is done, save the results back to the split stack
                self.split_stacks[i] = self.stack

        else:
             # otherwise just run the primitive
             primitive()

    
tile_engine = TileEngine()

# This worker is used for parallel execution in the multiprocessing engine    
def worker(input_list):
    
    rank = input_list[0]      # Rank
    iq = input_list[1]        # Input queue
    oq = input_list[2]        # Output queue
    primitive = input_list[3] # Primitive to run

    # Get the split bobs to process
    splitbobs = iq.get()

    ######################################################
    # FIXME: This seems to be having problems.
    # This code assumes that the data has not been read yet, it also assumes you have a 1 band raster
    # These assumptions don't always hold.
    tile = splitbobs[1]
    filehandle = gdal.Open(tile.filename)
    band = filehandle.GetRasterBand(1)
    reverse_rnum = filehandle.RasterYSize-tile.r-tile.nrows
    tile.data = band.ReadAsArray(tile.c,reverse_rnum,tile.ncols,tile.nrows)
    ######################################################
    
    ######FIX ME: Fetch vector data (does not work for now)########
    vector_data = []
    for bob in Config.inputs:
        if not isinstance(bob,Raster):
            vector_data.append(bob)
            break
        else:
            continue
    # Run the primitive on the splitbobs, record the output
    out = primitive(vector_data[0], tile)
    ######+++++++++++++++++++++++++++++++++++++++++++++++++########
    
    # Run the primitive on the splitbobs, record the output
    out = primitive(splitbobs[0], tile)
    
    ######################################################
    ## delete the tile.data before passing output 
    del tile
    tile = None
    ######################################################
                                     
    oq.put(out) # Save the output in the output queue

    return "worker %d %s" % (rank,splitbobs)    

# FIXME: Change to Engines.py    
class MultiprocessingEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(MultiprocessingEngine,self).__init__("MultiprocessingEngine")
        self.is_split=False
        
    def split(self, bobs):
        # Run the split from the TileEngine
        # That will provide a list of bobs in inputs to parallelize
        tile_engine.split(bobs)
        self.is_split = True
        
    # Merge (>)
    def merge(self, bobs):
        # Now that everything is merged set split to be false
        self.is_split = False

    # Sequence (==)
    def sequence(self, bobs):
        # If the Bobs are split, then handle it
        # If they are not, then there is nothing to do
        if self.is_split is True:
            # FIXME: Need to handle this
            print("NEED TO LOOP OVER SPLIT BOBS")
            pass # Loop over all the split Bobs
        pass

    # This method changes the run behavior to be in parallel.
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Get the inputs        
        inputs = Config.inputs
    
        # FIXME: REMOVE BELOW DATASTACK
        # Save the flows information in the global config data structure
        # FIXME: The problem with this solution is all data will be stored
        #        indefinitely, which is going to be a huge problem.
        # Config.flows[name] = {}
        # Config.flows[name]['input'] = inputs   
        # print(inputs)

        # If Bobs are not split, then it is easy
        if Config.engine.is_split is False:
            if isinstance(inputs,Bob):     # If it is a bob
                inputs = primitive(inputs)    # Just pass in the bob
            else:                          # If it is a list
                inputs = primitive(*inputs)   # De-reference the list and pass as parameters
        
        else: # When they are split we have to handle the list of Bobs and run in parallel

            # Make a pool of 4 processes
            # FIXME: THIS IS FIXED FOR NOW
            print("-> Number of processes = ", Config.n_core)

            pool = multiprocessing.Pool(Config.n_core)
            
            # Create a manager for the input and output queues (iq, oq)  
            m = multiprocessing.Manager()
            iq = m.Queue()
            oq = m.Queue()
            
            # Add split bobs to the input queue to be processed
            for splitbobs in inputs:
                iq.put(splitbobs)

            # How many times will we run the worker function using map
            mapsize = len(inputs)
            print(mapsize)

            # Make a list of ranks, queues, and primitives
            # These will be used for map_inputs
            ranklist = range(mapsize)
            iqlist = [iq for i in range(mapsize)]
            oqlist = [oq for i in range(mapsize)]
            prlist = [primitive for i in range(mapsize)]
        
            # Create map inputs by zipping the lists we just created
            map_inputs = zip(ranklist,iqlist,oqlist,prlist)
                       
            # Apply the inputs to the worker function using parallel map
            # Results can be printed for output from the worker tasks
            results = pool.map(worker, map_inputs)

            # Get the outputs from the output queue and save as new inputs
            inputs = []
            while not oq.empty():
                output = oq.get() # Get one output from the queue
                inputs.append(output) # Save to inputs
            
            # Done with the pool so close, then join (wait)
            pool.close()
            pool.join()

        # Save the outputs from this primitive
        # FIXME: REMOVE BELOW DATASTACK
        # Config.flows[name]['output'] = inputs
        
        # Save inputs from this/these primitive(s), for the next primitive
        if primitive.passthrough is False: # Typical case
            Config.inputs = inputs # Reset the inputs
        else:
            assert(Config.engine.is_split is False)
            Config.inputs.append(inputs) # Add to the inputs
            
        return inputs

    
mp_engine = MultiprocessingEngine()



# CUDA Engine
class CUDAEngine(Engine):

    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(CUDAEngine,self).__init__("CUDAEngine")
        self.is_split=False
        
    def split(self):
        # temp_stack = []
        # while self.stack:
        #      bob = stack.pop()
        #      gpu_bob = gpuarray.to_gpu(bob)
        #      temp_stack.push(gpu_bob)

        # while temp_stack: # Push it all back onto the stack maintaining order
        #     gpu_bob = temp_stack.pop()
        #     stack.push(gpu_bob)
        self.is_split = True
        
    # Merge (>)
    def merge(self):
        # Do the same thing as split, but in reverse. 
        # Pop everything off the stack and move from GPU to CPU memory

        # Now that everything is merged set split to be false
        self.is_split = False

    # Sequence (==)
    def sequence(self):
        # I don't think we need to do anything special in sequence for GPUs
        pass

    # This method will run a single primitive operation
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Check is_split, if running split then loop over split stacks
        if self.is_split:
            # Right now just run the primitive no matter what
            primitive()
        else:
            # otherwise just run the primitive
            primitive()

    # The rest should be fine for us right now.

cuda_engine = CUDAEngine()







# Set the Config.engine as the default

Config.engine = mp_engine
Config.engine = tile_engine
Config.engine = pass_engine
Config.engine = cuda_engine

print("Default engine",Config.engine)

if __name__ == '__main__':
    pass
