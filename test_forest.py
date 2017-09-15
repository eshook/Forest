"""
Copyright (c) 2017 Eric Shook. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
@author: eshook (Eric Shook, eshook@gmail.edu)
@contributors: <Contribute and add your name here!>
"""

# Begin coverage testing
import coverage
cov = coverage.coverage()
cov.start()

from forest import *
import unittest

# Import Forest unit tests
from unittests import *


# Test the full suite (each test file in the unittests directory stores a unittest suite)
# The following line collapses them into a 'full suite' of tests        
full_suite = unittest.TestSuite([test_Bob_suite,test_Bobs_suite,test_PrimitivesRaster_suite])

# Run the full suit using the 'unittest' package
unittest.TextTestRunner(verbosity=2).run(full_suite)

# Use this to test a single suite (e.g., test_Bob)
#unittest.TextTestRunner(verbosity=2).run(test_Bob_suite)

# Stop coverage testing and print out a report on percent of code covered by testing
cov.stop()
cov.report()
