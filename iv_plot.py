import fn

IV_SS_TIME_WINDOW = 0.02


def iv_plot(abf, show_plot=True):
    step_end = fn.get_step_boundaries(abf)[1]
    end_offset = fn.s_to_sample(IV_SS_TIME_WINDOW, abf)

    iv_data = fn.IVData(step_end - end_offset, step_end, True, abf)

    voltage_steps = iv_data.get_stimulus()
    current_response = iv_data.get_response(100)["filtered"][1]

    if show_plot:
        iv_plot = iv_data.plot()
        iv_plot.add_origin()
        iv_plot.set_labels("Time (s)", "Current (pA)", "Voltage (mV)", "Current (pA)")
        iv_plot.show()

    return {
        "voltage": voltage_steps,
        "current": current_response,
        "e_rev": fn.get_interp(voltage_steps, current_response, 0)[0],
    }


if __name__ == "__main__":
    import pyabf

    print(iv_plot(pyabf.ABF("abf/iv/iv_03.abf")))
