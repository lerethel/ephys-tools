import numpy as np

import fn

RM_TIME_WINDOW = 0.02


def rm(abf, show_plot=True):
    step_end = fn.get_step_boundaries(abf)[1]
    end_offset = fn.s_to_sample(RM_TIME_WINDOW, abf)

    current_steps = fn.get_iv_data(step_end - end_offset, step_end, 0, None, abf)[
        "original"
    ][1]
    voltage_response = fn.get_iv_data(step_end - end_offset, step_end, 1, 5, abf)

    m, b = np.polyfit(current_steps, voltage_response["filtered"][1], deg=1)
    line = np.polyval([m, b], current_steps)

    if show_plot:
        iv_plot = fn.plot_iv_data(
            current_steps, voltage_response, step_end - end_offset, step_end, 1, abf
        )

        iv_plot.subplots[1].plot(current_steps, line, color="red")
        iv_plot.subplots[3].plot(current_steps, line, color="red")
        iv_plot.set_labels("Time (s)", "Voltage (mV)", "Current (pA)", "Voltage (mV)")
        iv_plot.show()

    return m * 1000


if __name__ == "__main__":
    import pyabf

    print(rm(pyabf.ABF("abf/rm/rm_01.abf")))
