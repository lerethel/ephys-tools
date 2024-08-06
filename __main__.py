import glob
import pprint
import re
import traceback
from datetime import datetime
from os.path import dirname

import pyabf

from firing import firing
from first_ap import first_ap
from iv_plot import iv_plot
from rm import rm
from rmp import rmp
from sag_tau_cm import sag_tau_cm

DATA_FOLDER = "E:\\Записи обновленные"
RESULTS_FOLDER = f"{dirname(__file__)}\\results"
EXPERIMENT_DATE_FORMAT = r"%Y-%m-%d"


def create_protocol_re(*protocol_names):
    return re.compile(
        rf"(?:\d\s*-\s*)+(?:{'|'.join([re.escape(name) for name in protocol_names])})"
    )


def parse_ap_steps(*args):
    return {
        "RMP": rmp(*args),
        "First AP props": first_ap(*args),
        "Firing props": firing(*args),
    }


patterns = (
    (
        "Rm",
        create_protocol_re(
            "CC_Rm_from_-70mV",
            "CC_props_from_-70mV",
            "CC_steps_props",
        ),
        rm,
    ),
    (
        "Sag_tau_Cm",
        create_protocol_re(
            "CC_Ih_from_-70mV",
            "CC_Ih_from_RMP_and_-70mV",
            "CC_sag_from_-70mV",
            "CC_steps_Ih",
        ),
        sag_tau_cm,
    ),
    (
        "APs",
        create_protocol_re(
            "CC_AP_thrh",
            "CC_steps_from_RMP",
            "CC_steps_from_RMP_and_-70mV",
            "CC_steps_AP",
        ),
        parse_ap_steps,
    ),
    (
        "IV",
        create_protocol_re(
            "VC_IV_plot",
            "VC_standard",
            "VC_steps",
        ),
        iv_plot,
    ),
)


starting_date = datetime.strptime(
    input("Enter the experiment date from which to analyze cells (yyyy-mm-dd): "),
    EXPERIMENT_DATE_FORMAT,
)

all_patterns = [pattern[0] for pattern in patterns]
chosen_patterns = input(
    f"Choose patterns that will be used from {', '.join(all_patterns)}. "
    f"Separate by one whitespace. Type * to choose everything. "
)
chosen_patterns = chosen_patterns.split(" ") if chosen_patterns != "*" else all_patterns

show_plots = input('Show plots ("y" for "yes", anything for "no")? ').lower() == "y"

year_path = f"{DATA_FOLDER}\\{starting_date.year}\\"
r_cell_id = re.compile(rf".+?\\{starting_date.year}\\(.+?)\\\d+\.abf")
r_experiment_date = re.compile(rf".+?\\{starting_date.year}\\([^\\]+)")

results = {pattern: {} for pattern in chosen_patterns}

analysis_start_time = datetime.now()
evaluation_count = 0
exception_count = 0

for filepath in glob.iglob(f"{year_path}**\\*.abf", recursive=True):
    if (
        datetime.strptime(r_experiment_date.match(filepath)[1], EXPERIMENT_DATE_FORMAT)
        >= starting_date
    ):
        abf = pyabf.ABF(filepath, False)

        for pattern in patterns:
            if pattern[0] in results and pattern[1].match(abf.protocol):
                cell_id = r_cell_id.match(filepath)[1]

                if cell_id not in results[pattern[0]]:
                    results[pattern[0]][cell_id] = []

                try:
                    results[pattern[0]][cell_id].append(pattern[2](abf, show_plots))
                except Exception:
                    print(
                        f"\nException occurred with cell {cell_id}, file path: {filepath}\n"
                    )
                    print(traceback.format_exc())
                    exception_count += 1
                else:
                    evaluation_count += 1

print(
    f"Analysis finished in {(datetime.now() - analysis_start_time).total_seconds()} s. "
    f"Evaluations done: {evaluation_count}. "
    f"Exceptions occurred: {exception_count}"
)

results_for_file = []

for prop_name, prop_data in results.items():
    results_for_file.append(f"Property: {prop_name}")

    for cell_id, cell_data in prop_data.items():
        results_for_file.extend(
            (f"\nCell: {cell_id}", pprint.pformat(cell_data, sort_dicts=False))
        )

with open(
    f"{RESULTS_FOLDER}\\{str(datetime.now()).replace(":", "-")}.txt",
    "w",
) as file:
    file.write("\n".join(results_for_file))
