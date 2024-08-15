"""Checks if Firing.use() and FirstAP.use() return the same rheobases."""

import tests.t_fn as t_fn
from firing import Firing
from first_ap import FirstAP


def parse_firing_elem(results):
    rheobase_finder = (
        current_step
        for ap_num, current_step in zip(results["AP_numbers"], results["current_steps"])
        if ap_num
    )

    # Firing.use() counts AP numbers within a range of current steps and won't
    # include the rheobase if it's outside that range. Return None in such cases.
    return {"rheobase": next(rheobase_finder, None)}


def parse_first_ap_elem(results):
    return {"rheobase": results["rheobase"]}


class TestRheobase(t_fn.APTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_info = t_fn.build_test_info(Firing, parse_firing_elem)
        cls.test_info = t_fn.build_test_info(FirstAP, parse_first_ap_elem)

        for firing_elem, first_ap_elem in zip(cls.ref_info, cls.test_info):
            # Skip cases with no rheobase by making them always pass the test.
            if firing_elem["rheobase"] is None:
                firing_elem["rheobase"] = first_ap_elem["rheobase"]

    def test_rheobase_btw_methods(self):
        self.check_prop("rheobase")
