import tests.t_fn as t_fn
from firing import ANALYSIS_END_CURRENT_STEP, ANALYSIS_START_CURRENT_STEP, Firing

FIRING_REFERENCE_FILE_NAME = "firing_reference.csv"


def get_ref_info():
    return t_fn.build_ref_info(
        FIRING_REFERENCE_FILE_NAME, parse_ref_elem, filter_ref_elem
    )


def get_test_info():
    return t_fn.build_test_info(Firing, parse_test_elem, filter_test_elem)


def parse_ref_elem(row):
    return {
        "AP_numbers": list(map(int, row["AP_numbers"].split(" "))),
        "current_steps": list(map(int, row["current_steps"].split(" "))),
    }


def parse_test_elem(results):
    return {
        "AP_numbers": results["AP_numbers"],
        "current_steps": results["current_steps"],
    }


def filter_ref_elem(ref_elem):
    filtered_elem = {
        "filepath": ref_elem["filepath"],
        "AP_numbers": [],
        "current_steps": [],
    }

    for ap_num, current_step in zip(ref_elem["AP_numbers"], ref_elem["current_steps"]):
        if ANALYSIS_START_CURRENT_STEP <= current_step <= ANALYSIS_END_CURRENT_STEP:
            filtered_elem["AP_numbers"].append(ap_num)
            filtered_elem["current_steps"].append(current_step)

    return filtered_elem


def filter_test_elem(test_elem):
    filtered_elem = {
        "filepath": test_elem["filepath"],
        "AP_numbers": [],
        "current_steps": [],
    }

    for ap_num, current_step in zip(
        test_elem["AP_numbers"], test_elem["current_steps"]
    ):
        if ap_num:
            filtered_elem["AP_numbers"].append(ap_num)
            filtered_elem["current_steps"].append(current_step)

    return filtered_elem


class TestFiring(t_fn.APTest):
    @classmethod
    def setUpClass(cls):
        cls.ref_info = get_ref_info()
        cls.test_info = get_test_info()

    def test_firing(self):
        self.check_prop("AP_numbers")
        self.check_prop("current_steps")
