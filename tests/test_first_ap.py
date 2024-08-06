import tests.t_fn as t_fn
from first_ap import FirstAP

FIRST_AP_REFERENCE_FILE_NAME = "first_ap_reference.csv"

FIRST_AP_TEST_PARAM_MAP = {
    "rheobase": {"from_str_to": int, "delta": 0},
    **{
        param_name: {"from_str_to": float, "delta": 0.01}
        for param_name in (
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
        param_name: param_opts["from_str_to"](row[param_name])
        for param_name, param_opts in FIRST_AP_TEST_PARAM_MAP.items()
    }


def parse_test_elem(abf_results):
    return {
        param_name: abf_results[param_name] for param_name in FIRST_AP_TEST_PARAM_MAP
    }


class TestFirstAP(t_fn.APTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_info = get_ref_info()
        cls.test_info = get_test_info()

    def test_first_ap(self):
        for param_name, param_opts in FIRST_AP_TEST_PARAM_MAP.items():
            self.run_through(param_name, param_opts["delta"])
