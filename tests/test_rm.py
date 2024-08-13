import unittest

import pyabf

from rm import rm

RM_FILE_PATH = "abf\\rm\\rm_01.abf"
RM_REFERENCE_VALUE = 172.977
RM_DELTA = 5


class TestRm(unittest.TestCase):
    def test_rm(self):
        abf = pyabf.ABF(RM_FILE_PATH)
        self.assertAlmostEqual(rm(abf, False), RM_REFERENCE_VALUE, delta=RM_DELTA)
