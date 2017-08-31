"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: (Jacob Arndt, arndt204@umn.edu; )
"""

# FIXME: We need to have conditional imports here eventually
from collections import OrderedDict
from osgeo import ogr,gdal
import json
import fiona

from .Primitive import *
from ..bobs.Bobs import *

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
        
    def __call__(self, filename = None):    
            
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
            
        """Iterate through all features in the layer, getting their feature IDs, geometry, and attribute info.
        Write that information into the newLayer"""
        # FIXME: This loop needs to be optimized
        geom_types = [] # a list to hold the geometry types encountered in the shapefile.
        for feature in layer:
            FID = feature.GetFID()
            stringGeom = feature.GetGeometryRef().ExportToJson() #This export doesn't use tuples so this might not be the best idea. Tuples offer faster indexing.
            geom = json.loads(stringGeom) # convert the string representation of a dictionary to a real dictionary
            if geom['type'] not in geom_types:
                geom_types.append(geom['type'])
            attributes = OrderedDict()
            for key in fieldInfo.keys():
                attributes[key] = feature.GetField(key)
            newlayer[FID] = {"type": "Feature", "geometry": geom, "id": FID, "attributes": attributes}
        
        vector = Vector(miny,minx,maxy-miny,maxx-minx,None,None)
        vector.geom_types = geom_types
        vector.sr = spatialReference
        vector.data = newlayer
        return vector

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
        

class GeotiffReadPrim(Primitive):
    def __init__(self):
        # Call the __init__ for Primitive  
        super(GeotiffReadPrim,self).__init__("GeotiffRead")

        # Set passthrough to True so that data is passed through
        self.passthrough = True
        
    
    # FIXME: inbob is temporary for now, to solve a passthrough issue. Need to fix.
    def __call__(self, inbob = None, filename = None, bandnumber = 1):

        if filename is not None:
            self.filename = filename        

        ds = gdal.Open(self.filename) # Open gdal dataset
        if ds is None:
            raise PCMLException("Cannot open "+filename+" in ReadGeoTIFF")
    
        # By default get the first band
        band = ds.GetRasterBand(bandnumber)
        ncols = ds.RasterXSize
        nrows = ds.RasterYSize

        
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
        nparr=band.ReadAsArray(0,0,ncols,nrows) 
        layer.data = nparr
        #set_nparray(nparr,cellsize,nodata_value)
    
        del transform
        del nparr
        del band
        del ds
        nparr=None
        ds=None # Close gdal dataset
        band=None
    
        return layer

    def reg(self, filename):
        print(self.name,"register")
        self.filename = filename
        return self
    
GeotiffRead = GeotiffReadPrim()    
    
