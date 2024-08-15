import tests.t_fn as t_fn
from first_ap import FirstAP

FIRST_AP_REFERENCE_FILE_NAME = "first_ap_reference.csv"

FIRST_AP_TEST_PROPERTY_MAP = {
    "rheobase": {"from_str_to": int, "delta": 0},
    **{
        prop_name: {"from_str_to": float, "delta": 0.01}
        for prop_name in (
            "latency",
            "threshold",
            "amplitude",
            "half-width",
            "max_rise_slope",
            "max_decay_slope",
        )
    },
}


def get_ref_info():
    return t_fn.build_ref_info(FIRST_AP_REFERENCE_FILE_NAME, parse_ref_elem)


def get_test_info():
    return t_fn.build_test_info(FirstAP, parse_test_elem)


def parse_ref_elem(row):
    return {
        prop_name: prop_opts["from_str_to"](row[prop_name])
        for prop_name, prop_opts in FIRST_AP_TEST_PROPERTY_MAP.items()
    }


def parse_test_elem(results):
    return {prop_name: results[prop_name] for prop_name in FIRST_AP_TEST_PROPERTY_MAP}


class TestFirstAP(t_fn.APTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_info = get_ref_info()
        cls.test_info = get_test_info()

    def test_first_ap(self):
        for prop_name, prop_opts in FIRST_AP_TEST_PROPERTY_MAP.items():
            self.check_prop(prop_name, prop_opts["delta"])
