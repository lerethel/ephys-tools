"""Checks if Firing.use() and FirstAP.use() return the same rheobases."""

import tests.t_fn as t_fn
from firing import Firing
from first_ap import FirstAP


def parse_firing_info(abf_results):
    for ap_num, current_step in zip(
        abf_results["AP_numbers"], abf_results["current_steps"]
    ):
        if ap_num:
            return {"rheobase": current_step}

    # Firing.use() counts AP numbers within a range of current steps
    # and won't include the rheobase if it falls outside that range.
    return {"rheobase": None}


def parse_first_ap_info(abf_results):
    return {"rheobase": abf_results["rheobase"]}


class TestRheobase(t_fn.APTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_info = t_fn.build_test_info(Firing, parse_firing_info)
        cls.test_info = t_fn.build_test_info(FirstAP, parse_first_ap_info)

        for firing_elem, first_ap_elem in zip(cls.ref_info, cls.test_info):
            # Skip cases with no rheobase by making them always pass the test.
            if firing_elem["rheobase"] is None:
                firing_elem["rheobase"] = first_ap_elem["rheobase"]

    def test_rheobase_btw_methods(self):
        self.run_through("rheobase")
