"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

import rasterio
import rasterio.features
from collections import defaultdict
import numpy as np

from .Primitive import *
from ..bobs.Bobs import *

'''
TODO
1. Pass in __name__ rather than have it hard coded. More elegant.
2. Set name properly in super so it doesn't have to be duplicated.
'''

class PartialSumPrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(PartialSumPrim,self).__init__("PartialSum")

    def __call__(self, zone = None, data = None):

        # Create the key_value output bob
        out_kv = KeyValue(zone.h,zone.w,zone.y,zone.x)

        # Loop over the raster (RLayer)
        for r in range(len(data.data)):
            for c in range(len(data.data[0])):
                key = str(zone.data[r][c])
                if key in out_kv.data:
                    out_kv.data[key]['val'] += data.data[r][c]
                    out_kv.data[key]['cnt'] += 1
                else:
                    out_kv.data[key] = {}
                    out_kv.data[key]['val'] = data.data[r][c]
                    out_kv.data[key]['cnt'] = 1
        
        return out_kv

PartialSum = PartialSumPrim()

class AggregateSumPrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(AggregateSumPrim,self).__init__("AggregateSum")

    def __call__(self, *args):
        
        # Since it is an aggregator/reducer it takes in a list of bobs
        boblist = args
        
        # Set default values for miny,maxy,minx,maxx using first entry
        miny = maxy = boblist[0].y
        minx = maxx = boblist[0].x
        
        # Loop over bobs to find maximum spatial extent
        for bob in boblist:
            # Find miny,maxy,minx,maxx
            miny = min(miny,bob.y)
            maxy = max(maxy,bob.y)
            minx = min(minx,bob.x)
            maxx = max(maxx,bob.x)
        
        # Create the key_value output Bob that (spatially) spans all input bobs
        out_kv = KeyValue(miny, minx, maxy-miny, maxx-minx)

        # Set data to be an empty dictionary
        out_kv.data = {}
        
        # Loop over bobs, get keys and sum the values and counts
        for bob in boblist:
            # Loop over keys
            for key in bob.data:
                
                if key in out_kv.data:
                    out_kv.data[key]['val']+=bob.data[key]['val']
                    out_kv.data[key]['cnt']+=bob.data[key]['cnt']
                else:
                    out_kv.data[key] = {} # Create the entry and set val/cnt
                    out_kv.data[key]['val']=bob.data[key]['val']
                    out_kv.data[key]['cnt']=bob.data[key]['cnt']
                
        return out_kv

AggregateSum = AggregateSumPrim()


class AveragePrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(AveragePrim,self).__init__("Average")

    def __call__(self, sums = None):
        # Create the key_value output bob for average
        out_kv = KeyValue(sums.y, sums.x, sums.h, sums.w)

        for key in sums.data:
            out_kv.data[key] = float(sums.data[key]['val']) / float(sums.data[key]['cnt'])

        return out_kv

Average = AveragePrim()
        
# FIXME: Still in development.
class PartialSumRasterizePrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(PartialSumRasterizePrim,self).__init__("PartialSumRasterize")

    def __call__(self, zone = None, data = None, properties_name = None):

        # Create the transform for rasterio to rasterize the vector zones
        #print("bounds",data.x, data.y, data.x+data.w, data.y+data.h, data.cellsize, data.cellsize)
        transform = rasterio.transform.from_origin(data.x,data.y+data.h,data.cellsize,data.cellsize)
        
        properties_name = 'STATEFP' # or 'geoid'
        
        # Create zoneshapes, which is the geometry + state FP
        zoneshapes = ((f['geometry'],int(f['properties'][properties_name])) for f in zone.data)
        arr = rasterio.features.rasterize(shapes = zoneshapes, out_shape=data.data.shape, transform = transform)
        
        
        # TEMPORARY FOR LOOKING AT THE RESULTS
        if(False):
            with rasterio.open("examples/data/glc2000.tif") as src:
                profile = src.profile
                profile.update(count=1,compress='lzw')
                with rasterio.open('result.tif','w',**profile) as dst:
                    dst.write_band(1,arr)
            
            print("arr min=",np.min(arr))
            print("arr max=",np.max(arr))
            #print("arr avg=",np.avg(arr))
            print("arr shape",arr.shape)
        
        
        # Create the key_value output bob
        out_kv = KeyValue(zone.h,zone.w,zone.y,zone.x)

        print("Processing raster of size",data.nrows,"x",data.ncols)
        
        # Instead of looping over raster we can
        # zip zone[r] and data[r] to get key/value pairs
        # then we can apply for k,v in pairs: d[k] +=v
        # from : https://stackoverflow.com/questions/9285995/python-generator-expression-for-accumulating-dictionary-values
        # look here too : https://bugra.github.io/work/notes/2015-01-03/i-wish-i-knew-these-things-when-i-first-learned-python/
        # Loop over the raster (RLayer)
        '''
        for r in range(len(data.data)):
            for c in range(len(data.data[0])):
                key = str(arr[r][c])
                if key in out_kv.data:
                    out_kv.data[key]['val'] += data.data[r][c]
                    out_kv.data[key]['cnt'] += 1
                else:
                    out_kv.data[key] = {}
                    out_kv.data[key]['val'] = data.data[r][c]
                    out_kv.data[key]['cnt'] = 1
        '''
        
        #https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.unique.html#numpy.unique
        counts = np.unique(arr,return_counts=True)
        print("counts=",counts)
        
        # Loop over zone IDs
        for z in counts[0]:
            print("zoneid",z)
            
        # Create a dictionary from collections.defaultdict
        d=defaultdict(int)
        # Loop over the data and
        # Zip the zone keys (arr) and the data values into key,value pairs
        # Then add up the values from data and put into dictionary
        for r in range(len(data.data)):
            
            
            if(r%100==0):
                print("r=",r,"/",len(data.data))
            #Try 1, too slow    
            #kvzip = zip(arr[r],data.data[r])
            #for k,v in kvzip: d[k]+=v
            
            # Try 2, faster than Try 1, but still too slow.
            '''
            zonerow = arr[r]
            datarow = data.data[r]
            # Loop over unique zones
            for z in counts[0]:
                # This should set elements for zone z to 1, all others to 0
                zonemask = zonerow == z
                # Should zero out entries that are not the same as zone
                # So now you have an array of data elements that all belong to zone z
                datamask = datarow * zonemask
                # Add them all up and put them in the array
                d[z]+=np.sum(datamask)
            '''
        
        # Try 3, zonemask entire arrays (memory intensive, but faster)
        for z in counts[0]:
            print("z=",z)
            
            # This should set elements for zone z to 1, all others to 0
            zonemask = arr == z
            # Should zero out entries that are not the same as zone
                # So now you have an array of data elements that all belong to zone z
            datamask = data.data * zonemask
            # Add them all up and put them in the array
            d[z]+=np.sum(datamask)
                
        # Loop over d and counts to create output keyvalue bob                
        for i in range(len(counts[0])):
            countskey = counts[0][i]
            countscnt = counts[1][i]
            dsum = d[countskey]
            out_kv.data[countskey] = {}
            out_kv.data[countskey]['val'] = dsum
            out_kv.data[countskey]['cnt'] = countscnt
                    
        del arr

        return out_kv

PartialSumRasterize = PartialSumRasterizePrim()

class NearRepeatPrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(NearRepeatPrim,self).__init__("NearRepeat")
        
    #Config should contain global parameters for the primitive
    def __call__(self, bob):
        # Create the key_value output bob
        out_kv = KeyValue()
        # FIXME: Putting default values as this should be global
        distanceinterval=1000
        timeinterval=86400000
        maxdistance=5000
        maxtime=432000000
        timeranges=np.linspace(0,maxtime,num=(maxtime/timeinterval)+1,endpoint=True,dtype=np.int64)
        distanceranges=np.linspace(0,maxdistance,num=(maxdistance/distanceinterval)+1,endpoint=True)
        for i in xrange(len(timeranges)-1):
            for j in xrange(len(distanceranges)-1):
                out_kv.data[str(timeranges[i])+"-"+str(distanceranges[j])]={'val':0,'cnt':0}
        #First we calculate inter-distance and inter-time calculation for the bob 
        for i in xrange(len(bob.data)):
            for j in xrange(len(bob.data)):
                if i!=j:
                    timeshift=np.abs(bob.data[j]['t']-bob.data[i]['t'])
                    distanceshift= np.linalg.norm(np.asarray([bob.data[j]['x'],bob.data[j]['y']])-np.asarray([bob.data[i]['x'],bob.data[i]['y']]), 2, 0)
                    for ranges in out_kv.data:
                        timerange,distancerange=long(ranges.split('-')[0]),int(ranges.split('-')[1])
                        if timeshift>=timerange and timeshift<timerange+timeinterval and distanceshift>=distancerange and distanceshift<distancerange+distanceinterval:
                            out_kv.data[ranges]['cnt']+=1
        #since we are calculating pairs two times, need to divide results by 2
        for ranges in out_kv.data:
            out_kv.data[ranges]['cnt']/=2
        #Boundary calculation,since we have overlapping spatio temporal halo zones we have to avoid duplication
        for i in xrange(len(bob.halo)):
            for j in xrange(len(bob.data)):
                calculate=False
                #if the bob data is not from the halozone, then we could directly use it for calculations
                if bob.data[j]['x']>=bob.x+maxdistance and bob.data[j]['x']<bob.x+bob.w-maxdistance and bob.data[j]['y']>=bob.y+maxdistance and bob.data[j]['y']<bob.y+bob.h-maxdistance and bob.data[j]['t']>=bob.s+maxtime and bob.data[j]['t']<bob.s+bob.d-maxtime:
                    calculate=True
                #if the bob data is from an internal halo zone we only calculate the forward halozone positions 
                else:
                    if bob.halo[i]['x']>=bob.x and bob.halo[i]['x']<bob.x+bob.w+maxdistance and bob.halo[i]['y']>=bob.y and bob.halo[i]['y']<bob.y+bob.h+maxdistance and bob.halo[i]['t']>=bob.s and bob.halo[i]['t']<bob.s+bob.d+maxtime:
                        calculate=True
                if calculate:
                    timeshift=bob.data[j]['t']-bob.halo[i]['t']
                    distanceshift= np.linalg.norm(np.asarray([bob.data[j]['x'],bob.data[j]['y']])-np.asarray([bob.halo[i]['x'],bob.halo[i]['y']]), 2, 0)
                    for ranges in out_kv.data:
                        timerange,distancerange=long(ranges.split('-')[0]),int(ranges.split('-')[1])
                        if timeshift>=timerange and timeshift<timerange+timeinterval and distanceshift>=distancerange and distanceshift<distancerange+distanceinterval:
                            out_kv.data[ranges]['cnt']+=1
        return out_kv
    
NearRepeat = NearRepeatPrim()