
# Begin coverage testing
import coverage
cov = coverage.coverage(omit = "/usr/*")
cov.start()

from forest import *
import unittest

# Import Forest unit tests
from unittests import *

# Use this to test a single suite (e.g., test_Bob)
run_return = unittest.TextTestRunner(verbosity=2).run(test_Bmsb_suite)

# If unit tests are successful, then run a coverage test
if run_return.wasSuccessful():
    print("Unit tests successful, running coverage test")

    # Stop coverage testing and print out a report on percent of code covered by testing
    cov.stop()
    cov.report(show_missing = True)