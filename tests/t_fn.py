import csv
import glob
import os
import unittest

import pyabf

AP_FOLDER = os.path.join("abf", "aps")


def build_ref_info(ref_csv, parser, filter=None):
    ref_info = []

    with open(os.path.join(AP_FOLDER, ref_csv), "r") as ref_file:
        for row in csv.DictReader(ref_file):
            elem = {
                "filepath": os.path.join(AP_FOLDER, f"{row["filename"]}.abf"),
                **parser(row),
            }
            ref_info.append(filter(elem) if filter else elem)

    return ref_info


def build_test_info(test_cls, parser, filter=None):
    test_info = []

    for filepath in glob.iglob(os.path.join(AP_FOLDER, "*.abf")):
        abf = pyabf.ABF(filepath)
        elem = {
            "filepath": os.path.join(AP_FOLDER, f"{abf.abfID}.abf"),
            **parser(test_cls.use(abf, False)),
        }
        test_info.append(filter(elem) if filter else elem)

    return test_info


class APTest(unittest.TestCase):
    def check_prop(self, prop_name, delta=0):
        self.assertEqual(len(self.ref_info), len(self.test_info))

        for ref_elem, test_elem in zip(self.ref_info, self.test_info):
            with self.subTest(prop=prop_name, filepath=test_elem["filepath"]):
                if delta:
                    self.assertAlmostEqual(
                        test_elem[prop_name],
                        ref_elem[prop_name],
                        delta=delta,
                    )
                else:
                    self.assertEqual(test_elem[prop_name], ref_elem[prop_name])
