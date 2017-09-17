"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

from ..bobs.Bob import *
from ..bobs.Bobs import *
from . import Config
import math
import multiprocessing
import numpy as np

class Engine(object):
    def __init__(self, engine_type):
        self.engine_type = engine_type # Describes the type of engine
        self.is_split = False # The data are not split at the beginning
        
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

    # This method will run a single primitive operation
    # It will pull data from inputs and run the primitive
    # It will save the input
    def run(self, primitive):
        print("Running", primitive)

        # Get the name of the primitive operation being executed
        name = primitive.__class__.__name__

        # Get the inputs        
        inputs = Config.inputs
    
        # Save the flows information in the global config data structure
        # FIXME: The problem with this solution is all data will be stored
        #        indefinitely, which is going to be a huge problem.
        Config.flows[name] = {}
        Config.flows[name]['input'] = inputs   
    
        # If Bobs are not split, then it is easy
        if Config.engine.is_split is False:

            if isinstance(inputs,Bob):     # If it is a bob
                inputs = primitive(inputs)    # Just pass in the bob
            else:                          # If it is a list
                inputs = primitive(*inputs)   # De-reference the list and pass as parameters
        
        else: # When they are split we have to handle the list of Bobs
            new_inputs = []
            # Loop over the split bobs
            
            for splitbobs in inputs:
                out = None # Record output from primitive
                if isinstance(splitbobs,Bob): # If it is a bob
                    out = primitive(splitbobs)       # Just pass in the bob
                else:                         # If it is a list
                    out = primitive(*splitbobs)      # De-reference the list and pass as parameters
                new_inputs.append(out) # Save the output in the new_inputs list
            inputs = new_inputs
        
        # Save the outputs from this primitive
        Config.flows[name]['output'] = inputs
        
        # Save inputs from this/these primitive(s), for the next primitive
        if primitive.passthrough is False: # Typical case
            Config.inputs = inputs # Reset the inputs
        else:
            assert(Config.engine.is_split is False)
            Config.inputs.append(inputs) # Add to the inputs
            
        return inputs

# FIXME: Change to Engines.py    
class PassEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(PassEngine,self).__init__("PassEngine")
    
# This is the default engine that doesn't do anything.
pass_engine = PassEngine()    

# FIXME: Change to Engines.py    
class TileEngine(Engine):
    def __init__(self):
        # FIXME: Need an object to describe type of engines rather than a string
        super(TileEngine,self).__init__("TileEngine")
        self.is_split=False
        
    # Split (<)
    
    # FIXME: Split also has to reach into Config.flows in case if functions pull out of list
    # Keeping a nested open/bound variable stack might be easier than flows
    # Think this one through
    
    def split(self, bobs):
        
        # Set the number of tiles to split to
        # FIXME: Eventually this should be determined or user-defined.
        num_tiles = 2
        
        # If already split, do nothing.
        if self.is_split is True:
            return
        
        new_inputs = []
        # Loop over bobs in inputs to split
        for bob in Config.inputs:
            
            # For each bob, create a new split tile (TileEngine :)
            tiles = []
            
            # Split only works for rasters for now
            # For all other data types (e.g., vectors) we just duplicate the data
            if not isinstance(bob,Raster): # Check if not a raster
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
                tile.datatype = bob.datatype
                # FIXME: Need a better method to copy these over.
                
                # Split the data (depends on raster/vector)
                tile.data = bob.get_data(tile_r,tile_c,tile_nrows,tile_ncols)
                                
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

    
tile_engine = TileEngine()

# This worker is used for parallel execution in the multiprocessing engine    
def worker(input_list):
    
    rank = input_list[0]      # Rank
    iq = input_list[1]        # Input queue
    oq = input_list[2]        # Output queue
    primitive = input_list[3] # Primitive to run

    # Get the split bobs to process
    splitbobs = iq.get()

    # Run the primitive on the splitbobs, record the output
    out = primitive(*splitbobs)

    oq.put(out) # Save the output in the output queue

    return "worker %d %s" % (rank,splitbobs) # Can be printed if needed
    
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
    
        # Save the flows information in the global config data structure
        # FIXME: The problem with this solution is all data will be stored
        #        indefinitely, which is going to be a huge problem.
        Config.flows[name] = {}
        Config.flows[name]['input'] = inputs   

        #import pdb; pdb.set_trace()


        # If Bobs are not split, then it is easy
        if Config.engine.is_split is False:
            if isinstance(inputs,Bob):     # If it is a bob
                inputs = primitive(inputs)    # Just pass in the bob
            else:                          # If it is a list
                inputs = primitive(*inputs)   # De-reference the list and pass as parameters
        
        else: # When they are split we have to handle the list of Bobs and run in parallel

            # Make a pool of 4 processes
            # FIXME: THIS IS FIXED FOR NOW
            pool = multiprocessing.Pool(4)
            
            # Create a manager for the input and output queues (iq, oq)            
            m = multiprocessing.Manager()
            iq = m.Queue()
            oq = m.Queue()
            
            # Add split bobs to the input queue to be processed
            for splitbobs in inputs:
                iq.put(splitbobs)

            # How many times will we run the worker function using map
            mapsize = len(inputs)
            
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
        Config.flows[name]['output'] = inputs
        
        # Save inputs from this/these primitive(s), for the next primitive
        if primitive.passthrough is False: # Typical case
            Config.inputs = inputs # Reset the inputs
        else:
            assert(Config.engine.is_split is False)
            Config.inputs.append(inputs) # Add to the inputs
            
        return inputs

    
mp_engine = MultiprocessingEngine()



# Set the Config.engine as the default

Config.engine = pass_engine
Config.engine = tile_engine
Config.engine = mp_engine


print("Default engine",Config.engine)

if __name__ == '__main__':
    pass