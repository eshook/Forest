"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""


from forest import *


r1 = Raster()
r2 = Raster()

r1.data[0][0]=10
r1.data[3][3]=20


print(r1)
print(r1.data)

r2.data[0][0]=1
r2.data[0][1]=2
r2.data[0][2]=3
r2.data[0][3]=4

print(r2)
print(r2.data)

ro = LocalSum(r1,r2)

print(ro)
print(ro.data)

print("Add")
o = r1+r2

print(o)
print(o.data)

"Add together"
o2 = ro + o

print("o2",o2)
print(o2.data)

sub = ro - o
print("sub",sub)
print(sub.data)

div = ro / o

print("div",div)
print(div.data)

mul = ro * o
print("mul",mul)
print(mul.data)

#glc = ReadGeoTiff("examples/data/GLOBCOVER_2004.tif")

#out = run_primitive(GeoTiffRead.reg("examples/data/GLOBCOVER_2004.tif") == LocalSum)



