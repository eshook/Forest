"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Jacob Arndt, arndt204@umn.edu; Luyi Hunter, chen3461@umn.edu; Xinran Duan, duanx138@umn.edu)
"""

# FIXME: We need to have conditional imports here eventually
from collections import OrderedDict
from osgeo import ogr,gdal,osr
import json
import fiona
import csv
from .Primitive import *
from ..bobs.Bobs import *
import numpy as np
from dateutil.parser import parse
import time

class RasterDataTestPrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(RasterDataTestPrim,self).__init__("RasterRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    def __call__(self, filename = None):
        # FIXME: Ignoring filename right now. Hardcoding raster dataset
        if filename is not None:
            self.filename = filename        

        # Create raster data structure
        raster = Raster()
        
        # FIXME: Hardcoded for now
        
        # Set "values" for a raster dataset
        for r in range(len(raster.data)):
            for c in range(len(raster.data[0])):
                raster.data[r][c] = r+c

        return raster

    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
        
RasterDataTest = RasterDataTestPrim()


class VectorZoneTestPrim(Primitive):
    def __init__(self):

        # Call the __init__ for Primitive  
        super(VectorZoneTestPrim,self).__init__("VectorZoneTest")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
        
    def __call__(self, filename = None):
        # FIXME: Ignoring filename right now. Hardcoding raster dataset
        if filename is not None:
            self.filename = filename        

        # Create vector data structure
        #vector = Vector()

        # FIXME: For now we are hardcoding a "rasterized" vector data
        rasterizedvector = Raster()
        
        # Set "zones" for a "rasterized" vector dataset
        for r in range(len(rasterizedvector.data)):
            for c in range(len(rasterizedvector.data[0])):
                rasterizedvector.data[r][c] = c

        return rasterizedvector

    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
        
VectorZoneTest = VectorZoneTestPrim()


# Original author: Jacob Arndt
class ShapefileReadPrim(Primitive):
    def __init__(self):
        
        # Call the __init__ for Primitive  
        super(ShapefileReadPrim,self).__init__("ShapefileRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    def __call__(self, filename = None, others = None):    
            
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(filename, 0)
        layer = dataSource.GetLayer()
        layerDefinition = layer.GetLayerDefn()
        fieldInfo = OrderedDict()# the fields are ordered for consistency between features.
        # A layer is an unordered set of features described by type, geometry, id, and attributes. Each feature's
        # key is a unique FID.  
        newlayer = {} 
        spatialReference = str(layer.GetSpatialRef())
        
        """Get y,x reference from the first feature to use for initializing the Bob and its origin for this vector layer"""
        firstFeature = layer[0]
        envelope = firstFeature.GetGeometryRef().GetEnvelope()
        miny = envelope[2]
        minx = envelope[0]
        maxy = envelope[3]
        maxx = envelope[1]
            
        """get all field names, widths, and precisions for the shapefile and store them in an ordered dictionary.
        they will be used in defining the Vector Bob's Table...?"""
        for field in range(layerDefinition.GetFieldCount()):
            fieldName =  layerDefinition.GetFieldDefn(field).GetName()
            fieldTypeCode = layerDefinition.GetFieldDefn(field).GetType()
            fieldType = layerDefinition.GetFieldDefn(field).GetFieldTypeName(fieldTypeCode)
            fieldWidth = layerDefinition.GetFieldDefn(field).GetWidth()
            fieldPrecision = layerDefinition.GetFieldDefn(field).GetPrecision()
            fieldInfo[fieldName] = [fieldType,fieldWidth,fieldPrecision]

        #Right now, manually finding the boundaries of the BOB. Need to see where
        #these values are stored inside of the shapefile and just access them there
        #Mckenzie
        miny = float('inf')
        minx = float('inf')
        maxy = float('-inf')
        maxx = float('-inf')
        mins = float('inf')
        maxs = float('-inf')
            
        """Iterate through all features in the layer, getting their feature IDs, geometry, and attribute info.
        Write that information into the newLayer"""
        # FIXME: This loop needs to be optimized
        geom_types = [] # a list to hold the geometry types encountered in the shapefile.
        for feature in layer:
            FID = feature.GetFID()
            stringGeom = feature.GetGeometryRef().ExportToJson() #This export doesn't use tuples so this might not be the best idea. Tuples offer faster indexing.
            geom = json.loads(stringGeom) # convert the string representation of a dictionary to a real dictionary

            #Finds the boundary coordinates for the vector
            if geom["coordinates"][0] < minx:
                minx = geom["coordinates"][0]
            elif geom["coordinates"][0] > maxx:
                maxx = geom["coordinates"][0]

            if geom["coordinates"][1] < miny:
                miny = geom["coordinates"][1]
            elif geom["coordinates"][1] > maxy:
                maxy = geom["coordinates"][1]

            if "time" in attributes:
                if attributes["time"] > maxs:
                    maxs = attributes["time"]
                elif attributes["time"] < mins:
                    mins = attributes["time"]


            if geom['type'] not in geom_types:
                geom_types.append(geom['type'])
            attributes = OrderedDict()
            for key in fieldInfo.keys():
                attributes[key] = feature.GetField(key)
            newlayer[FID] = {"type": "Feature", "geometry": geom, "id": FID, "attributes": attributes}

        #Checks if the shapefile contained any information on time (what is the
        #standard attribute name for time attributes? Is there one?)
        if "time" in newlayer[0]["attributes"]:
            vector = Vector(miny,minx,maxy-miny,maxx-minx,mins,maxs-mins)
        else:
            vector = Vector(miny,minx,maxy-miny,maxx-minx,None,None)
            
        vector.geom_types = geom_types
        vector.sr = spatialReference
        vector.data = newlayer
        #Currently returns others in order to create better data flow throughout
        #the primitives. Should all data be passed between primitives, or is there
        #a central location to "put" objects, and how would a primitive know
        #which data sets to grab?
        return [vector, others]

    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
        
ShapefileRead = ShapefileReadPrim()   


# Original author: Jacob Arndt
class ShapefileNewReadPrim(Primitive):
    def __init__(self):
        
        # Call the __init__ for Primitive  
        super(ShapefileNewReadPrim,self).__init__("ShapefileNewRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    def __call__(self, filename = None):    
        
        if filename is not None:
            self.filename = filename
        
        crs = None
        features = []
        y = x = h = w = 0.0
        with fiona.collection(self.filename) as shp:
            print("Boundingbox",shp.bounds)
            for feature in shp:
                features.append(feature)
            y = shp.bounds[0]
            x = shp.bounds[1]
            h = shp.bounds[2] - y
            w = shp.bounds[3] - x
                
        vector = Vector(y, x, h, w)
        vector.data = features
        return vector
    
    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
    
ShapefileNewRead = ShapefileNewReadPrim()
        

# FIXME: Rename GeotiffRead to RasterTileRead
class GeotiffReadPrim(Primitive):
    def __init__(self):
        # Call the __init__ for Primitive  
        super(GeotiffReadPrim,self).__init__("GeotiffRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    
    # FIXME: inbob is temporary for now, to solve a passthrough issue. Need to fix.
    def __call__(self, inbob = None, filename = None, bandnumber = 1, paralell = False):

        if filename is not None:
            self.filename = filename        

        ds = gdal.Open(self.filename) # Open gdal dataset
        if ds is None:
            raise PCMLException("Cannot open "+filename+" in ReadGeoTIFF")
    
        # By default get the first band
        band = ds.GetRasterBand(bandnumber)
        ncols = ds.RasterXSize
        nrows = ds.RasterYSize

        # nrows = 10000

        if band is None:
            print("Cannot read selected band in "+filename+" in ReadGeoTIFF")
            raise(Exception)
            
        nodata_value = band.GetNoDataValue()
        if nodata_value is None:
            nodata_value=-9999
        transform = ds.GetGeoTransform()
        cellsize = transform[1]
        origin_x = transform[0]
        origin_y = transform[3]
        x=origin_x
        y=origin_y-nrows*cellsize
        if(abs(transform[1])!=abs(transform[5])): # pixelwidth=1, pixelheight=5
            PCMLUserInformation("Cells of different height and width selecting width not height")
    
        h=float(nrows)*cellsize
        w=float(ncols)*cellsize
        layer=Raster(y,x,h,w,None,None,nrows,ncols,cellsize)
        
        ######################################################
        ## Enable paralell processing
        paralell = True 
        if paralell == False:
            nparr=band.ReadAsArray(0,0,ncols,nrows) 
            layer.data = nparr
        layer.filename = self.filename
        layer.nodatavalue = nodata_value
        ######################################################
        # set_nparray(nparr,cellsize,nodata_value)
        del transform
        del band
        del ds
        transform=None
        ds=None 
        band=None
        ######################################################
        if paralell == False:
            del nparr
            nparr=None
        ######################################################
    
        return layer

    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
    
GeotiffRead = GeotiffReadPrim()    


#Primitive to read a CSV file and generate 2D point or Spatio-temporal point layer
class CsvReadPrim(Primitive):
    def __init__(self):
        
        # Call the __init__ for Primitive  
        super(CsvReadPrim,self).__init__("CsvRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    def __call__(self, filename = None):
        if filename is not None:
            self.filename=filename
        with open(self.filename) as csvfile:
            # we need coordinates in meters for distance based calculations and decomposition
            source = osr.SpatialReference()
            #always expecting data to be un-projected
            source.ImportFromEPSG(4326)
            target = osr.SpatialReference()
            #web mercator projection in meters
            target.ImportFromEPSG(3857)
            transform = osr.CoordinateTransformation(source,target)
            reader = csv.DictReader(csvfile)
            #for calculating bounds
            minx,miny=np.inf,np.inf
            maxx,maxy=np.NINF,np.NINF
            mint,maxt=np.inf,np.NINF
            data=[]
            isspatiotemporal=False
            for row in reader:
                #this mapping should come from global parameters
                lat,lon=float(row['y']),float(row['x'])
                timeval=None
                #this mapping should come from global parameters
                if 't' in row:
                    timeval=row['t']
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(lon,lat)
                point.Transform(transform)
                if point.GetX()<minx:
                    minx=point.GetX()
                if point.GetY()<miny:
                    miny=point.GetY()
                if point.GetX()>maxx:
                    maxx=point.GetX()
                if point.GetY()>maxy:
                    maxy=point.GetY()
                pointdat={}
                pointdat['x']=point.GetX()
                pointdat['y']=point.GetY()
                if timeval is not None:
                    #parse to handle lot of time formats
                    timedat=parse(timeval, fuzzy=True)
                    #time should be in milli for easier calculations
                    timeinfo=long(time.mktime(timedat.timetuple())*1000 + timedat.microsecond/1000)
                    if timeinfo<mint:
                        mint=timeinfo
                    if timeinfo>maxt:
                        maxt=timeinfo
                    pointdat['t']=timeinfo
                    isspatiotemporal=True
                data.append(pointdat)
        y,x,h,w,s,d = miny,minx,maxy-miny,maxx-minx,mint,maxt-mint
        layer=None
        if isspatiotemporal:
            layer=STPoint(y,x,h,w,s,d)
        else:
            layer=Point(y,x,h,w,s,d)
        layer.data=data
        return layer
    
    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self

CsvRead=CsvReadPrim()    


#Primitive used to create .tif files to store output data in
#Mckenzie Ebert
class GeotiffWritePrim(Primitive):

    def __init__(self):
        super(GeotiffWritePrim, self).__init__("GeoTIFF Write Prim")
        self.name = "GeoTIFF Write Primitive"


    def __call__(self, rasterBob, spatialReference, filePath = None):
        
        #File will be saved to current working directory if no file path is defined
        if filePath == None:
            filePath = str(os.getcwd())+"/TestResults"

        #Adds the name of the file to be created to the file path
        filePath = filePath + "/Results" + str(time.strftime("%m")) + "-" + str(time.strftime("%d"))+".tif"
        driver = gdal.GetDriverByName('GTiff') #Obtains the GeoTIFF Driver

        #Creates a new .tif file at the indicated file path, with an xSize of w and a ySize of h
        newGTiff = driver.Create(filePath, rasterBob.ncols, rasterBob.nrows, 1, gdal.GDT_Float32)
        band = newGTiff.GetRasterBand(1)
        band.WriteArray(rasterBob.data) #Writes the data from the raster to the new .tif file
        band.SetNoDataValue(-1) #Can be changed to whatever value fits the circumstance (still need to decide on the default value)
        

        #Sets up the GeoTransform and the Projection for the file
        xRes = rasterBob.w/float(rasterBob.ncols)
        yRes = rasterBob.h/float(rasterBob.nrows)
        geoTransform = (rasterBob.x, xRes, 0, rasterBob.y, 0, yRes)

        newGTiff.SetGeoTransform(geoTransform)
        newGTiff.SetProjection(spatialReference.ExportToWkt())

        band.FlushCache() #'Saves' the data written to the file

        #Closes out the file and raster band
        band = None
        newGTiff = None
        
        return

GeotiffWrite = GeotiffWritePrim()

#Edited from the original CsvReadPrim in order to get a Space Time Cube BOB - is there a better way to optimize for this type of BOB? Or does this work?
class CsvSTCReadPrim(Primitive):
    def __init__(self):
        
        # Call the __init__ for Primitive  
        super(CsvSTCReadPrim,self).__init__("CsvSTCRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    def __call__(self, filename = None):
        if filename is not None:
            self.filename=filename
        with open(self.filename) as csvfile:
            # we need coordinates in meters for distance based calculations and decomposition
            source = osr.SpatialReference()
            #always expecting data to be un-projected
            source.ImportFromEPSG(4326)
            target = osr.SpatialReference()
            #web mercator projection in meters
            target.ImportFromEPSG(3857)
            transform = osr.CoordinateTransformation(source,target)
            reader = csv.DictReader(csvfile)
            #for calculating bounds
            minx,miny=np.inf,np.inf
            maxx,maxy=np.NINF,np.NINF
            mint,maxt=np.inf,np.NINF
            data=[]
            isspatiotemporal=False
            for row in reader:
                #this mapping should come from global parameters
                lat,lon=float(row['y']),float(row['x'])
                timeval=None
                #this mapping should come from global parameters
                if 't' in row:
                    timeval=row['t']
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(lon,lat)
                point.Transform(transform)
                if point.GetX()<minx:
                    minx=point.GetX()
                if point.GetY()<miny:
                    miny=point.GetY()
                if point.GetX()>maxx:
                    maxx=point.GetX()
                if point.GetY()>maxy:
                    maxy=point.GetY()
                pointdat={}
                pointdat['x']=point.GetX()
                pointdat['y']=point.GetY()
                if timeval is not None:
                    #parse to handle lot of time formats
                    timedat=parse(timeval, fuzzy=True)
                    #time should be in milli for easier calculations
                    timeinfo=long(time.mktime(timedat.timetuple())*1000 + timedat.microsecond/1000)
                    if timeinfo<mint:
                        mint=timeinfo
                    if timeinfo>maxt:
                        maxt=timeinfo
                    pointdat['t']=timeinfo
                    isspatiotemporal=True
                data.append(pointdat)
        y,x,h,w,s,d = miny,minx,maxy-miny,maxx-minx,mint,maxt-mint
        layer=None
        if isspatiotemporal:
            layer=STPoint(y,x,h,w,s,d)
        else:
            layer=Point(y,x,h,w,s,d)
        layer.data=data
        return layer
    
    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self

CsvSTCRead=CsvSTCReadPrim()


class GeoTIFFMultiWriter(Primitive):

    def __init__(self):
        super(GeoTIFFMultiWriter, self).__init__("GeoTIFF Multi-Writer")
        self.name = "GeoTIFF Multi-Writer"

    def __call__(self, STCube, spatialRef, filePath = None):
        for time in STCube.timelist:
            
            #File will be saved to current working directory if no file path is defined
            if filePath == None:
                filePath = str(os.getcwd())+"/TestResults-"+str(time)

            #Adds the name of the file to be created to the file path
            filePath = filePath + "/Results" + str(time.strftime("%m")) + "-" + str(time.strftime("%d"))+"-"+str(time)+".tif"
            driver = gdal.GetDriverByName('GTiff') #Obtains the GeoTIFF Driver

            #Creates a new .tif file at the indicated file path, with an xSize of w and a ySize of h
            newGTiff = driver.Create(filePath, rasterBob.ncols, rasterBob.nrows, 1, gdal.GDT_Float32)
            band = newGTiff.GetRasterBand(1)
            band.WriteArray(rasterBob.data) #Writes the data from the raster to the new .tif file
            band.SetNoDataValue(-1) #Can be changed to whatever value fits the circumstance (still need to decide on the default value)

            #Sets up the GeoTransform and the Projection for the file
            xRes = rasterBob.w/float(rasterBob.ncols)
            yRes = rasterBob.h/float(rasterBob.nrows)
            geoTransform = (rasterBob.x, xRes, 0, rasterBob.y, 0, yRes)

            newGTiff.SetGeoTransform(geoTransform)
            newGTiff.SetProjection(spatialReference.ExportToWkt())

            band.FlushCache() #'Saves' the data written to the file

            #Closes out the file and raster band
            band = None
            newGTiff = None
            
        return

multiGeoWriter = GeoTIFFMultiWriter()
