from forest import *
from forest.bobs.Bobs import *

import forest.engines.Config

# Debugger line
#import pdb; pdb.set_trace()


def testit_randomstuff():
    # Make an empty Bob
    b = Bob()
    print(b)

    # Make an empty raster dataset
    r = Raster(0,0,20,20,10,10,2)
    print(r)

def zonalaverage_oldschool(zonefilename, datafilename):
    
    # Read a rasterized Vector dataset as zones
    # FIXME: For now it does not read the file    
    vector = VectorZoneTest(zonefilename)
    print("vector=",vector)
    print(vector.data)

    # Read a raster dataset as data
    # FIXME: For now it does not read the file    
    raster = RasterDataTest(datafilename)    
    print("raster=",raster)
    print(raster.data)

    # Calculate partial sum of vector and raster
    ps = PartialSum(vector, raster)
    print("PartialSum=",ps)
    print(ps.data)    

    zonalaverage = Average(ps)

    return zonalaverage
    
def testit_oldschool(zonefilename, datafilename):

    print("starting old school")

    zonalaverage = zonalaverage_oldschool(zonefilename, datafilename)
    
    print("ZonalAverage=",zonalaverage)
    
    for zone in sorted(zonalaverage.data):
        print(zone,"=",zonalaverage.data[zone])
    
    print("finished old school")

    
def zonalaverage_forest(zonefilename, datafilename):

    output = run_primitive( VectorZoneTest.reg(zonefilename) == RasterDataTest.reg(datafilename) < PartialSum > AggregateSum == Average )

    return output
    
    
def testit_forest(zonefilename, datafilename):
    print("starting forest")
    
    zonalaverage = zonalaverage_forest(zonefilename, datafilename)
    
    print("ZonalAverage=",zonalaverage)
    
    for zone in sorted(zonalaverage.data):
        print(zone,"=",zonalaverage.data[zone])
    
    print("finished forest")


    
if __name__ == '__main__':
    
    zonefilename = "examples/data/vector.shp"
    datafilename = "examples/data/raster.tif"
    
        
    #testit_oldschool(zonefilename,datafilename)
    
    testit_forest(zonefilename,datafilename)
