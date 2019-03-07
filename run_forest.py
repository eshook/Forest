from forest import *
from forest.bobs.Bobs import *

import forest.engines.Config

# Debugger line
#import pdb; pdb.set_trace()


# Demonstrates how to program the traditional way using forest functions
def zonalaverage_oldschool(zonefilename, datafilename):
    
    # Read a rasterized Vector dataset as zones
    # FIXME: For now it does not read the file    
    vector = ShapefileNewRead(zonefilename)
    print("vector=",vector)
    print("data len=",len(vector.data))

    # Read a raster dataset as data
    # FIXME: For now it does not read the file    
    raster = GeotiffRead(filename = datafilename)    
    print("raster=",raster)
    print("raster data len",len(raster.data))
    print("raster data sum",sum(raster.data.flatten()))


    # Calculate partial sum of vector and raster
    ps = PartialSumRasterize(vector, raster)
    print("PartialSum=",ps)
    print(ps.data)    

    zonalaverage = Average(ps)

    return zonalaverage
   
# Test the old school way 
def testit_oldschool(zonefilename, datafilename):

    print("starting old school")

    zonalaverage = zonalaverage_oldschool(zonefilename, datafilename)
    
    print("ZonalAverage=",zonalaverage)
    
    for zone in sorted(zonalaverage.data):
        print(zone,"=",zonalaverage.data[zone])
    
    print("finished old school")

# Demonstrates how to program the 'forest' way 
def zonalaverage_forest(zonefilename, datafilename):

    #output = run_primitive( VectorZoneTest.reg(zonefilename) == RasterDataTest.reg(datafilename) < PartialSum > AggregateSum == Average )

    output = run_primitive( ShapefileNewRead.reg(zonefilename) == GeotiffRead.reg(datafilename) < PartialSumRasterize > AggregateSum == Average )
    #output = run_primitive( ShapefileNewRead.reg(zonefilename) == GeotiffRead.reg(datafilename) == PartialSumRasterize )

    return output
    
# Test the forest way
def testit_forest(zonefilename, datafilename):
    print("starting forest")
    
    zonalaverage = zonalaverage_forest(zonefilename, datafilename)
    
    print("ZonalAverage=",zonalaverage)
    print("ZAData      =",zonalaverage.data)
    for zone in sorted(zonalaverage.data):
        print(zone,"=",zonalaverage.data[zone])
    
    print("finished forest")

# Try using the new NearRepeat calculator
def nearrepeat_forest(datafilename):
    output = run_primitive(CsvRead.reg(datafilename)< NearRepeat > AggregateSum)
    return output
   

# Test the near repeat calculator
def testit_nearrepeat(datafile):
    print("staring near repeat test")

    nr = nearrepeat_forest(datafile)

    print("Near Repeat = ", nr)

    print("finished near repeat test")
 
if __name__ == '__main__':
    #print("Uncomment tests below once data is in examples/data")    
    zonefilename = "examples/data/states.shp"
    datafilename = "examples/data/glc2000.tif"
    #datafilename = "examples/data/crimes.csv"
    
        
    #testit_oldschool(zonefilename,datafilename)
    testit_forest(zonefilename,datafilename)
    #testit_nearrepeat(datafile)

