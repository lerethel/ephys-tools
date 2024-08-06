import fn

IV_SS_TIME_WINDOW = 0.02


def iv_plot(abf, show_plot=True):
    step_end = fn.get_step_boundaries(abf)[1]
    end_offset = fn.s_to_sample(IV_SS_TIME_WINDOW, abf)

    voltage_steps = fn.get_iv_data(step_end - end_offset, step_end, 1, None, abf)[
        "original"
    ][1]
    current_response = fn.get_iv_data(step_end - end_offset, step_end, 0, 100, abf)

    if show_plot:
        iv_plot = fn.plot_iv_data(
            voltage_steps, current_response, step_end - end_offset, step_end, 0, abf
        )

        iv_plot.add_origin()
        iv_plot.set_labels("Time (s)", "Current (pA)", "Voltage (mV)", "Current (pA)")
        iv_plot.show()

    return {"voltage": voltage_steps, "current": current_response["filtered"][1]}


if __name__ == "__main__":
    import pyabf

    print(iv_plot(pyabf.ABF("abf/iv/iv_03.abf")))
