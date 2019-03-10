"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Load forest
from forest import *

# Create two raster datasets
r1 = Raster()
r2 = Raster()

# Modify 2 cells in raster 1
r1.data[0][0]=10
r1.data[3][3]=20

# Print out raster 1 and it's data
print(r1)
print(r1.data)

# Modify a few cells in raster 2
r2.data[0][0]=1
r2.data[0][1]=2
r2.data[0][2]=3
r2.data[0][3]=4

# Print out raster 2 and it's data
print(r2)
print(r2.data)

# Apply local sum and print out the results
#ro = LocalSum(r1,r2) # FIXME: LocalSum is no longer an option in the new rendition of Forest
ro = r1+r2

print(ro)
print(ro.data)

# Try adding them and printing out the results (hint they should be identical)
print("Add")
o = r1+r2

print(o)
print(o.data)

# Try adding the two results together and print them out
"Add together"
o2 = ro + o

print("o2",o2)
print(o2.data)


# Subtract the first result from the second result, see what you get.
sub = ro - o
print("sub",sub)
print(sub.data)

# Now let's try to divide them. Notice we should get some not-a-number (nan) results.
# This is due to the fact that o2 has some zeros.
div = ro / o2

print("div",div)
print(div.data)

# What about multiply? Yay it works too!
mul = ro * o
print("mul",mul)
print(mul.data)



