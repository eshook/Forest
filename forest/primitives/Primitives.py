"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

import rasterio
import rasterio.features
from collections import defaultdict


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

    def __call__(self, zone = None, data = None):

        #arr = np.zeros((data.nrows,data.ncols))
        
        print("type=",type(zone.data))
        #for k in zone.data:
        #    print("  key=",k)
        
        print("data0=",data.data[0])
        
        #            0=xmin,3=ymax, 1=pixel width, 5=pixel height, 2=line width, 4=line width
        # We might want -cellsize for 5
        #transform = [data.x,data.cellsize,0,data.y+data.h,0,-data.cellsize]
        transform = rasterio.transform.from_origin(data.x,data.y+data.h,data.cellsize,data.cellsize)
        
        # out_shape = (data.nrows,data.ncols)
        #arr = rasterio.features.rasterize(shapes = [data.data], out_shape=(data.nrows,data.ncols), transform = transform)
        
        print("outshape",data.data.shape)
        print("transform",transform)
        
        # Kinda working
        #arr = rasterio.features.rasterize(shapes = zone.data, out_shape=data.data.shape, transform = transform)

        #print("first entry",zone.data[0]['geometry'])
        
        # FIRST ELEMENT WORKS!
        #arr = rasterio.features.rasterize(shapes = [ (zone.data[0]['geometry'],int(zone.data[0]['properties']['STATEFP'])) ], out_shape=data.data.shape, transform = transform)
        
        zoneshapes = ((f['geometry'],int(f['properties']['STATEFP'])) for f in zone.data)
        zonearr = rasterio.features.rasterize(shapes = zoneshapes, out_shape=data.data.shape, transform = transform)
        
        '''
        shapes = []
        for f in zone.data:
            shapes.append([ f['geometry'],f['properties']['STATEFP'] ])
        
        #shapes = ((geom,value) for geom, value in zip(zone.data[])
        
        zoneshapes = ((f['geometry'],f['properties']['STATEFP']) for f in zone.data)
        
        print("zoneshapes[0]=",zoneshapes[0])
        
        arr = rasterio.features.rasterize(shapes = zoneshapes, out_shape=data.data.shape, transform = transform)
        '''
        
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
        
        print("first entry in arr",zonearr[0][0])
        
        
        # Create the key_value output bob
        out_kv = KeyValue(zone.h,zone.w,zone.y,zone.x)

        print("Processing raster of size",data.nrows,"x",data.ncols)
        
#         # Instead of looping over raster we can
#         # zip zone[r] and data[r] to get key/value pairs
#         # then we can apply for k,v in pairs: d[k] +=v
#         # from : https://stackoverflow.com/questions/9285995/python-generator-expression-for-accumulating-dictionary-values
#         # look here too : https://bugra.github.io/work/notes/2015-01-03/i-wish-i-knew-these-things-when-i-first-learned-python/
#         # Loop over the raster (RLayer)
#         '''
#         for r in range(len(data.data)):
#             for c in range(len(data.data[0])):
#                 key = str(arr[r][c])
#                 if key in out_kv.data:
#                     out_kv.data[key]['val'] += data.data[r][c]
#                     out_kv.data[key]['cnt'] += 1
#                 else:
#                     out_kv.data[key] = {}
#                     out_kv.data[key]['val'] = data.data[r][c]
#                     out_kv.data[key]['cnt'] = 1
#         '''
        
#         #https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.unique.html#numpy.unique
#         counts = np.unique(arr,return_counts=True)
#         print("counts=",counts)
        
#         # Loop over zone IDs
#         for z in counts[0]:
#             print("zoneid",z)
            
#         # Create a dictionary from collections.defaultdict
#         d=defaultdict(int)
#         # Loop over the data and
#         # Zip the zone keys (arr) and the data values into key,value pairs
#         # Then add up the values from data and put into dictionary
#         for r in range(len(data.data)):
            
            
#             if(r%100==0):
#                 print("r=",r,"/",len(data.data))
#             #Try 1, too slow    
#             #kvzip = zip(arr[r],data.data[r])
#             #for k,v in kvzip: d[k]+=v
            
#             # Try 2, faster than Try 1, but still too slow.
#             '''
#             zonerow = arr[r]
#             datarow = data.data[r]
#             # Loop over unique zones
#             for z in counts[0]:
#                 # This should set elements for zone z to 1, all others to 0
#                 zonemask = zonerow == z
#                 # Should zero out entries that are not the same as zone
#                 # So now you have an array of data elements that all belong to zone z
#                 datamask = datarow * zonemask
#                 # Add them all up and put them in the array
#                 d[z]+=np.sum(datamask)
#             '''
        
#         # Try 3, zonemask entire arrays (memory intensive, but faster)
#         for z in counts[0]:
#             print("z=",z)
            
#             # This should set elements for zone z to 1, all others to 0
#             zonemask = arr == z
#             # Should zero out entries that are not the same as zone
#                 # So now you have an array of data elements that all belong to zone z
#             datamask = data.data * zonemask
#             # Add them all up and put them in the array
#             d[z]+=np.sum(datamask)
                
                
#         print("d=",d)
        
#         for i in range(len(counts[0])):
#             countskey = counts[0][i]
#             countscnt = counts[1][i]
#             dsum = d[countskey]
#             out_kv.data[countskey] = {}
#             out_kv.data[countskey]['val'] = dsum
#             out_kv.data[countskey]['cnt'] = countscnt
            
        # Try 4 np.bincount with np.unique
        
        zonearr_flat = zonearr.flatten()

        # Bottle-neck 1. np.unique
        # Consider doing only once for a time series of the requested area
        zonereal,zonereal_counts = np.unique(zonearr, return_counts = True)
        dict_count = dict(zip(zonereal, zonereal_counts.T))
        
        # Create a dummy zone id list to match those dummy zone sums created by bincount
        zonedummy = list(range(zonereal.min(),zonereal.max()+1))
        
        # Conduct Zonal analysis
        # Bottle-neck 2. np.bincount
        zonedummy_sums = np.bincount(zonearr_flat, weights=data.data.flatten())
        
        print("Output Length: ", len(zonedummy_sums))
        print(zonedummy_sums)
        print("Dummy Zone Length: ", len(zonedummy))
        print(zonedummy)
        print("Real Zone Length: ", len(zonereal))
        print(zonereal, zonereal_counts)
        
        # Zip zone ids with valid zone sums and zone counts into a dictionary
        dict_sum = dict(zip(zonedummy, zonedummy_sums.T))
        dict_count = dict(zip(zonereal, zonereal_counts.T))
        for zoneid in zonereal:
            out_kv.data[zoneid] = {}
            out_kv.data[zoneid]['val'] = dict_sum[zoneid]
            out_kv.data[zoneid]['cnt'] = dict_count[zoneid]
        print(out_kv)
        
        
        # Try 5 np.bincount with pandas.'unique'
        
        # zonearr_flat = zonearr.flatten()
        #
        # # pandas 'unique'
        # import pandas as pd
        # zone_ss = pd.Series(zonearr_flat)
        # dict_count = zone_ss.value_counts().to_dict()
        # zonereal = list(dict_count.keys())
        #
        # # Create a dummy zone id list to match those dummy zone sums created by bincount
        # zonedummy = list(range(int(min(zonereal)),int(max(zonereal))+1))
        #
        # # Conduct Zonal analysis
        # # Bottle-neck 2. np.bincount
        # # zone_df = pd.DataFrame({'index': zonearr_flat, 'value': zonearr_flat})
        # # zone_group = zone_df.groupby(['index'], sort=False).sum()
        # # print(dict(list(zone_group)))
        # zonedummy_sums = np.bincount(zonearr_flat, weights=data.data.flatten())
        #
        # print("Output Length: ", len(zonedummy_sums))
        # print(zonedummy_sums)
        # print("Dummy Zone Length: ", len(zonedummy))
        # print(zonedummy)
        # print("Real Zone Length: ", len(zonereal))
        #
        # # Zip zone ids with valid zone sums into a dictionary
        # dict_sum = dict(zip(zonedummy, zonedummy_sums.T))
        # # Zip values and counts into small dict and put them into the Bob
        # for zoneid in zonereal:
        #     out_kv.data[zoneid] = {}
        #     out_kv.data[zoneid]['val'] = dict_sum[zoneid]
        #     out_kv.data[zoneid]['cnt'] = dict_count[zoneid]
        # print(out_kv)
        
        del zonearr

        return out_kv

PartialSumRasterize = PartialSumRasterizePrim()
