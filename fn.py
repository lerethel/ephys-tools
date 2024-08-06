import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FIGURE_INIT_PARAMS = {"figsize": (10, 6), "layout": "constrained"}
DISTANCE_BETWEEN_MARKERS_AND_MAX_PEAK = 10

AP_MV_THRESHOLD = -20
AP_MV_MS_THRESHOLD = 10

FILTER_MIN_DISTANCE_FROM_LAST_SUPERTHRESHOLD_VALUE = 5
IV_TRACE_TITLE = "%s signal"
IV_CURVE_TITLE = "I/V curve for %s signal"

AP_MARKER_STYLE = {
    "marker": "d",
    "color": "purple",
}

IV_CURVE_ORIGIN_STYLE = {"color": "black", "linewidth": 0.5}
IV_TRACE_ANALYZED_SPAN_STYLE = {
    "alpha": 0.2,
    "facecolor": "purple",
}

# https://numpy.org/devdocs/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
np.set_printoptions(legacy="1.25")

######################
### COMMON METHODS ###


def sample_to_s(sample_no, abf):
    return sample_no / abf.sampleRate


def s_to_sample(s, abf):
    return round(s * abf.sampleRate)


def get_closest(context, target_value):
    index = np.abs(context - target_value).argmin()
    return index, context[index]


def get_prepulse(step_start, abf, offset_s):
    prepulse_i = step_start - s_to_sample(offset_s, abf)
    return prepulse_i if prepulse_i >= 0 else 0


def get_postpulse(step_end, abf, offset_s):
    postpulse_i = step_end + s_to_sample(offset_s, abf)
    return postpulse_i if postpulse_i < abf.sweepPointCount else abf.sweepPointCount - 1


def get_step_boundaries(abf):
    for sweep_no in abf.sweepList:
        abf.setSweep(sweep_no, channel=0)

        stimulus = abf.sweepC
        step_indexes = np.where(stimulus != abf.holdingCommand[0])[0]

        if len(step_indexes):
            step_start = step_indexes[0]
            step_end = step_indexes[-1]

            return step_start, step_end


def get_current_step(abf):
    stimulus = abf.sweepC
    step_indexes = np.where(stimulus != abf.holdingCommand[0])[0]

    # Call round() because a float close to the stimulus strength can be returned in rare cases.
    return round(
        (stimulus[step_indexes[0]] - stimulus[0]) * 10 if len(step_indexes) else 0
    )


# Unlike abf.sweepDerivative, this function
# (1) doesn't shift the sample indexes relative to abf.sweepY, and
# (2) converts the calculated values to dy/ms rather than dy/s.
def get_derivative(start_index, end_index, abf):
    dy_dt = np.diff(abf.sweepY[start_index:end_index])
    dy_dt = np.insert(dy_dt, 0, [dy_dt[0]])
    dy_dt *= abf.sampleRate / 1000
    return dy_dt


########################################
### COMMON METHODS FOR PLOTTING DATA ###


def set_all_xylabels(x, y):
    for ax in plt.gcf().get_axes():
        ax.set_xlabel(x)
        ax.set_ylabel(y)


def move_figure(figure, x, y):
    backend = matplotlib.get_backend().lower()

    if backend == "tkagg":
        figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == "wxagg":
        figure.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK.
        figure.canvas.manager.window.move(x, y)


def show_plot(abf):
    figure = plt.gcf()

    all_handles = []
    all_labels = []

    for ax in figure.get_axes():
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels

    if len(all_labels):
        by_label = dict(zip(all_labels, all_handles))

        figure.legend(
            by_label.values(),
            by_label.keys(),
            loc="outside lower center",
            ncol=len(all_labels),
        )

    move_figure(figure, 0, 0)
    plt.suptitle(abf.abfFilePath, weight="bold")
    plt.show()


def extend_coords(start_index, start_offset, end_index, end_offset, abf):
    prepulse = get_prepulse(start_index, abf, start_offset)
    postpulse = get_postpulse(end_index, abf, end_offset)

    return abf.sweepX[prepulse:postpulse], abf.sweepY[prepulse:postpulse]


################
### IV PLOTS ###


def filter_signal(start_index, end_index, y_ms_threshold, abf):
    filtered_signal = np.copy(abf.sweepY[start_index:end_index])

    abs_context_derivative = np.abs(get_derivative(start_index, end_index, abf))
    y_above_trh_indexes = np.where(abs_context_derivative >= y_ms_threshold)[0]

    if len(y_above_trh_indexes):
        # In case there's a spike crossing the left border of the analysis window.
        if y_above_trh_indexes[0] == 0:
            filtered_signal[0] = np.mean(filtered_signal)
        # In case there's a spike crossing the right border of the analysis window.
        if y_above_trh_indexes[-1] == len(abs_context_derivative) - 1:
            y_above_trh_indexes = y_above_trh_indexes[:-1]

    prev_i = -FILTER_MIN_DISTANCE_FROM_LAST_SUPERTHRESHOLD_VALUE

    for cur_i in y_above_trh_indexes:
        filtered_signal[cur_i] = filtered_signal[cur_i - 1]

        if cur_i - prev_i < FILTER_MIN_DISTANCE_FROM_LAST_SUPERTHRESHOLD_VALUE:
            filtered_signal[prev_i + 1 : cur_i + 1] = filtered_signal[prev_i]

        prev_i = cur_i

    return filtered_signal


def get_iv_data(start_index, end_index, channel, filter_threshold, abf):
    original_signal_per_sweep = []
    original_mean_signal_per_sweep = []

    filtered_signal_per_sweep = []
    filtered_mean_signal_per_sweep = []

    for sweep_no in abf.sweepList:
        abf.setSweep(sweep_no, channel)

        original_signal_per_sweep.append(abf.sweepY[start_index:end_index])
        original_mean_signal_per_sweep.append(np.mean(original_signal_per_sweep[-1]))

        if filter_threshold is not None:
            filtered_signal_per_sweep.append(
                filter_signal(start_index, end_index, filter_threshold, abf)
            )
            filtered_mean_signal_per_sweep.append(
                np.mean(filtered_signal_per_sweep[-1])
            )

    return {
        "original": (original_signal_per_sweep, original_mean_signal_per_sweep),
        "filtered": (filtered_signal_per_sweep, filtered_mean_signal_per_sweep),
    }


def plot_iv_data(x_data, y_data, start_index, end_index, channel, abf):
    plt.figure(**FIGURE_INIT_PARAMS)

    subplots = []

    for response_type in y_data.keys():
        subplots.append(plt.subplot(2, 2, len(subplots) + 1))
        plt.title(IV_TRACE_TITLE % response_type.capitalize())

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, channel)

            cur_y = abf.sweepY

            if response_type == "filtered":
                cur_y = np.copy(abf.sweepY)
                cur_y[start_index:end_index] = y_data[response_type][0][sweep_no]

            plt.plot(*extend_coords(start_index, 0.02, end_index, 0.02, abf))

        plt.axvspan(
            sample_to_s(start_index, abf),
            sample_to_s(end_index, abf),
            **IV_TRACE_ANALYZED_SPAN_STYLE,
        )

        subplots.append(plt.subplot(2, 2, len(subplots) + 1))
        plt.title(IV_CURVE_TITLE % response_type)
        plt.scatter(x_data, y_data[response_type][1])
        plt.plot(x_data, y_data[response_type][1])

    return IVPlot(subplots, abf)


class IVPlot:
    def __init__(self, subplots, abf):
        self.subplots = subplots
        self.abf = abf

    def add_origin(self):
        for i in (1, 3):
            self.subplots[i].axhline(0, **IV_CURVE_ORIGIN_STYLE)
            self.subplots[i].axvline(0, **IV_CURVE_ORIGIN_STYLE)

    def set_labels(self, x_trace, y_trace, x_curve, y_curve):
        for i, subplot in enumerate(self.subplots):
            subplot.set_xlabel(x_trace if i % 2 == 0 else x_curve)
            subplot.set_ylabel(y_trace if i % 2 == 0 else y_curve)

    def show(self):
        show_plot(self.abf)


#########################
### ACTION POTENTIALS ###


def is_ap(index, context_derivative, points_to_average=5):
    return (
        context_derivative[index] >= AP_MV_MS_THRESHOLD
        and np.mean(np.abs(context_derivative[index : index + points_to_average]))
        >= AP_MV_MS_THRESHOLD
    )


def find_ap_peaks(start_index, end_index, abf):
    peak_indexes = []

    context_voltages = abf.sweepY[start_index:end_index]
    context_derivative = get_derivative(start_index, end_index, abf)

    search_indexes = np.where(
        # Filter out subthreshold mV values and zero mV/ms values as useless.
        (context_voltages >= AP_MV_THRESHOLD) & (context_derivative != 0)
    )[0]

    prev_mv_ms = float("-inf")
    rising_phase = False

    for i in search_indexes:
        cur_mv_ms = context_derivative[i]

        if not rising_phase and is_ap(i, context_derivative):
            rising_phase = True

        if rising_phase and cur_mv_ms < 0 and prev_mv_ms > 0:
            peak_i = i - 1 + start_index
            peak_indexes.append(peak_i)
            rising_phase = False

        prev_mv_ms = cur_mv_ms

    return peak_indexes
