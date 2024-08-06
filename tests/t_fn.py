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

            if filter:
                elem = filter(elem)

            ref_info.append(elem)

    return ref_info


def build_test_info(test_cls, parser, filter=None):
    test_info = []

    for filepath in glob.iglob(os.path.join(AP_FOLDER, "*.abf")):
        abf = pyabf.ABF(filepath)

        elem = {
            "filepath": os.path.join(AP_FOLDER, f"{abf.abfID}.abf"),
            **parser(test_cls.use(abf, False)),
        }

        if filter:
            elem = filter(elem)

        test_info.append(elem)

    return test_info


class APTest(unittest.TestCase):
    def run_through(self, param_name, delta=0):
        self.assertEqual(len(self.ref_info), len(self.test_info))

        for ref_elem, test_elem in zip(self.ref_info, self.test_info):
            with self.subTest(param=param_name, filepath=test_elem["filepath"]):
                if delta:
                    self.assertAlmostEqual(
                        test_elem[param_name],
                        ref_elem[param_name],
                        delta=delta,
                    )
                else:
                    self.assertEqual(test_elem[param_name], ref_elem[param_name])
