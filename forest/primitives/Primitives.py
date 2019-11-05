"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Luyi Hunter, chen3461@umn.edu; Xinran Duan, duanx138@umn.edu)
@contributors: <Contribute and add your name here!>
"""

import rasterio
import rasterio.features
from collections import defaultdict
import numpy as np
import pandas as pd

from ..core.Primitive import *
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

#Mckenzie Ebert
class VectorToSTCubePrim(Primitive):

    def __init__(self):
        #Call the __init__ for Primitive
        super(VectorToSTCubePrim, self).__init__("Change Vector BOB into a STCube BOB")

    def __call__(self, vector, attrName, others):

        cellHeight = int(vector.h/1000)
        cellWidth = int(vector.w/1000)
        #By putting the points into this format, it is possible to use a KDTree to speed up the cell assigning process for point values
        pointList, pointValues, timeList = vector.getSTCPoints(attrName)
        #Setting up initial values and dimensions
        stCube = STCube(vector.y, vector.x, vector.h, vector.w, vector.s, vector.d)
        stCube.nrows = 1000
        stCube.ncols = 1000
        stCube.cellwidth = cellWidth
        stCube.cellheight = cellHeight
        stCube.timelist = timeList
        stCube.srid = vector.sr
        stCube.setdata()

        #The maximum distance a point can be away from the center of a raster cell and still be inside the cell
        maxDist = ((cellWidth/2)**2+(cellHeight/2)**2)**0.5
        pointTree = sp.cKDTree(pointList)

        for r,c in stCube.iterrc():
            yVal, xVal = stCube.findCellCenter(r, c)

            #This returns a circular radius. Points will still need to be checked if they are within the rectangular raster cell
            closePoints = pointTree.query_ball_point([xVal,yVal], maxDist)
            
            for tIndex in range(len(timeList)):                
                time = timeList[tIndex]
                numPoints = 0
                for point in closePoints:
                    xCoor = pointList[point][0]
                    yCoor = pointList[point][1]

                    if pointValues[point][1] == time and abs(xCoor-xVal)<=(cellWidth/2) and abs(yCoor-yVal)<=(cellHeight/2):
                        stCube.data[tIndex][r][c] += pointValues[point][0]
                        numPoints+=1

                if numPoints != 0:
                    stCube.data[tIndex][r][c] = (stCube.data[tIndex][r][c])/numPoints
                                                                                
        return [stCube, others]

VectorToSTCube = VectorToSTCubePrim()


        
# FIXME: Still in development.
class PartialSumRasterizePrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(PartialSumRasterizePrim,self).__init__("PartialSumRasterize")

    def __call__(self, zone = None, data = None, properties_name = None):

        #print("type=",type(zone.data))
        
        #print("data0=",data.data[0])
        
        #            0=xmin,3=ymax, 1=pixel width, 5=pixel height, 2=line width, 4=line width
        # We might want -cellsize for 5
        #transform = [data.x,data.cellsize,0,data.y+data.h,0,-data.cellsize]
        transform = rasterio.transform.from_origin(data.x,data.y+data.h,data.cellsize,data.cellsize)
        
        # out_shape = (data.nrows,data.ncols)
        #arr = rasterio.features.rasterize(shapes = [data.data], out_shape=(data.nrows,data.ncols), transform = transform)
        
        #print("outshape",data.data.shape)
        #print("transform",transform)
        
	# Pull out the zone geometries with the state code
        zoneshapes = ((f['geometry'],int(f['properties']['STATEFP'])) for f in zone.data)

        # zoneshapes = ((f['geometry'],int(f['properties']['geoid'])) for f in zone.data)
        zonearr = rasterio.features.rasterize(shapes = zoneshapes, out_shape=data.data.shape, transform = transform)

#         # TEMPORARY FOR LOOKING AT THE RESULTS
#         if(False):
#             with rasterio.open("examples/data/glc2000.tif") as src:
#                 profile = src.profile
#                 profile.update(count=1,compress='lzw')
#                 with rasterio.open('result.tif','w',**profile) as dst:
#                     dst.write_band(1,arr)
            
#             print("arr min=",np.min(arr))
#             print("arr max=",np.max(arr))
#             #print("arr avg=",np.avg(arr))
#             print("arr shape",arr.shape)
        
#         print("first entry in arr",zonearr[0][0])
        
        
        # Create the key_value output bob
        out_kv = KeyValue(zone.h,zone.w,zone.y,zone.x)

        print("Processing raster of size",data.nrows,"x",data.ncols)
        
        # Try 5 np.bincount with pandas.'unique' -> This was the fastest
        # For Tries 1-4 look at github history. :)
        
        zonearr_flat = zonearr.flatten()
        value_flat = data.data.flatten()

        print("valueflatsum=",sum(value_flat))

        empty_value = np.amax(zonearr_flat)+2
        #zonearr_flat[value_flat < (data.nodatavalue+1)] = empty_value # TEMPORARY FIXME: BRING BACK?
        
        # pandas 'unique'
        zone_ss = pd.Series(zonearr_flat)
        #print("zone_ss",zone_ss)

        # Zip values and counts into small dict and put them into the Bob
        dict_count = zone_ss.value_counts().to_dict()
        #print("dict_count",dict_count,"empty_value",empty_value)
        #del dict_count[empty_value] # TEMPORARY FIXME: BRING BACK?


        if len(dict_count) < 1:
            pass
        else:
            zonereal = list(dict_count.keys())

            # Create a dummy zone id list to match those dummy zone sums created by bincount
            zonedummy = list(range(int(min(zonereal)),int(max(zonereal))+1))

            # Conduct Zonal analysis
            zonedummy_sums = np.bincount(zonearr_flat, weights=data.data.flatten())

            print("zaf: ", zonearr_flat)
            print("wddf:", data.data.flatten())
            print("Zonedummy sums: ", zonedummy_sums)
            print("Zonedummy sums: ", zonedummy_sums)
            #print("Output Length: ", len(zonedummy_sums))
            #print("Dummy Zone Length: ", len(zonedummy))
            #print("Real Zone Length: ", len(zonereal))

            # Zip zone ids with valid zone sums into a dictionary
            dict_sum = dict(zip(zonedummy, zonedummy_sums.T))
            print("dict_sum",dict_sum)

            for zoneid in zonereal:
                out_kv.data[zoneid] = {}
                out_kv.data[zoneid]['val'] = dict_sum[zoneid]
                out_kv.data[zoneid]['cnt'] = dict_count[zoneid]
            print(out_kv)
        
        del zonearr
        zonearr = None

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
                #if the bob data is not from the interior halozone, then we could ignore it
                if bob.data[j]['x']>=(bob.x+maxdistance) and bob.data[j]['x']<(bob.x+bob.w-maxdistance) and bob.data[j]['y']>=(bob.y+maxdistance) and bob.data[j]['y']<(bob.y+bob.h-maxdistance) and bob.data[j]['t']>=(bob.s+maxtime) and bob.data[j]['t']<(bob.s+bob.d-maxtime):
                    calculate=False
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


class vectorToSTCubePrim(Primitive):

    def __init__(self):

        # Call the __init__ for Primitive  
        super(vectorToSTCubePrim,self).__init__("Vector to STCube")

    def __call__(self, vector, attrName, others):

        stCube = STCube(vector.y, vector.x, vector.h, vector.w, vector.s, vector.d)
        stCube.srid = vector.sr #Check that this is correct

        coords, values, timeList = vector.getSTCPoints(attrName)
        stCube.timelist = timeList
        #Assuming regular time intervals
        stCube.cellwidth = timeList[1]-timeList[0]
        
        for time in timeList:
            layer = np.zeros(())
        










        
