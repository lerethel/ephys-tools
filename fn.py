import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FIGURE_INIT_PARAMS = {"figsize": (10, 6), "layout": "constrained"}
DISTANCE_BETWEEN_MARKERS_AND_MAX_PEAK = 10

AP_MIN_AMPLITUDE = 5
AP_MV_THRESHOLD = -25
AP_MV_MS_RISE_THRESHOLD = 10
AP_MV_MS_FALL_THRESHOLD = -5
AP_MV_MS_RISE_MIN_REPEAT_DURATION = 0.0005
AP_MV_MS_FALL_MIN_REPEAT_DURATION = 0.0005
AP_STEP_EXTENSION = 0.005

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


def get_voltage_step(abf):
    stimulus = abf.sweepC
    step_indexes = np.where(stimulus != abf.holdingCommand[0])[0]

    return stimulus[step_indexes[0]] if len(step_indexes) else abf.holdingCommand[0]


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


class IVData:
    def __init__(self, start_index, end_index, voltage_clamp, abf):
        self.abf = abf
        self.start_index = start_index
        self.end_index = end_index
        self.voltage_clamp = voltage_clamp
        self.response_channel = 0 if voltage_clamp else 1

    def _filter_signal(self, y_ms_threshold):
        abf = self.abf
        start_index = self.start_index
        end_index = self.end_index

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

    def get_stimulus(self):
        abf = self.abf

        stimulus_sweeps = []

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, channel=0)
            stimulus_sweeps.append(
                get_voltage_step(abf) if self.voltage_clamp else get_current_step(abf)
            )

        self.stimulus = stimulus_sweeps
        return stimulus_sweeps

    def get_response(self, filter_threshold):
        abf = self.abf

        original_signal_per_sweep = []
        original_mean_signal_per_sweep = []

        filtered_signal_per_sweep = []
        filtered_mean_signal_per_sweep = []

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, self.response_channel)

            original_signal_per_sweep.append(
                abf.sweepY[self.start_index : self.end_index]
            )
            original_mean_signal_per_sweep.append(
                np.mean(original_signal_per_sweep[-1])
            )

            if filter_threshold is not None:
                filtered_signal_per_sweep.append(self._filter_signal(filter_threshold))
                filtered_mean_signal_per_sweep.append(
                    np.mean(filtered_signal_per_sweep[-1])
                )

        self.response = {
            "original": (original_signal_per_sweep, original_mean_signal_per_sweep),
            "filtered": (filtered_signal_per_sweep, filtered_mean_signal_per_sweep),
        }

        return self.response

    def plot(self):
        abf = self.abf
        x_data = self.stimulus
        y_data = self.response
        start_index = self.start_index
        end_index = self.end_index

        plt.figure(**FIGURE_INIT_PARAMS)

        subplots = []

        for response_type in y_data.keys():
            subplots.append(plt.subplot(2, 2, len(subplots) + 1))
            plt.title(IV_TRACE_TITLE % response_type.capitalize())

            for sweep_no in abf.sweepList:
                abf.setSweep(sweep_no, self.response_channel)

                cur_y = abf.sweepY

                if response_type == "filtered":
                    cur_y = np.copy(abf.sweepY)
                    cur_y[start_index:end_index] = y_data[response_type][0][sweep_no]

                prepulse = get_prepulse(start_index, abf, 0.02)
                postpulse = get_postpulse(end_index, abf, 0.02)
                plt.plot(abf.sweepX[prepulse:postpulse], cur_y[prepulse:postpulse])

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


def find_aps(start_index, end_index, abf):
    # Extend the analysis window slightly beyond the step to catch APs that might be on its edge.
    original_end_index = end_index
    end_index = get_postpulse(end_index, abf, AP_STEP_EXTENSION)

    context_voltages = abf.sweepY[start_index:end_index]
    context_derivative = get_derivative(start_index, end_index, abf)
    mv_ms_above_trh_indexes = np.where(context_derivative >= AP_MV_MS_RISE_THRESHOLD)[0]
    mv_ms_rise_points_to_check = s_to_sample(AP_MV_MS_RISE_MIN_REPEAT_DURATION, abf)
    mv_ms_fall_points_to_check = s_to_sample(AP_MV_MS_FALL_MIN_REPEAT_DURATION, abf)

    # Put consecutive indexes with values above the mV/ms threshold into separate lists.
    # The first value in each list will be the threshold of a possible AP.
    events = np.split(
        mv_ms_above_trh_indexes, np.where(np.diff(mv_ms_above_trh_indexes) != 1)[0] + 1
    )

    peak_indexes = []
    trh_indexes = []

    for i, indexes in enumerate(events):
        # Consider an event a possible AP if its mV/ms values stay superthreshold long enough.
        if len(indexes) >= mv_ms_rise_points_to_check:
            cur_trh_i = indexes[0]

            # Ignore APs that occurred outside the step.
            if cur_trh_i + start_index >= original_end_index:
                break

            try:
                next_trh_i = events[i + 1][0]
            except IndexError:
                # This is to find the last peak since nothing goes after it.
                next_trh_i = len(context_derivative) - 1

            # Some APs seem to have a phase where the voltage rises steeply
            # and then slowly and can even slighly fall before rising steeply again.
            # This might happen when too much current is injected into a cell.
            # Filter out such phases and accept only events with a fairly steep falling phase.
            fall_trh_indexes = np.where(
                context_derivative[cur_trh_i:next_trh_i] <= AP_MV_MS_FALL_THRESHOLD
            )[0]

            if len(fall_trh_indexes) >= mv_ms_fall_points_to_check:
                fall_trh_i = fall_trh_indexes[0] + cur_trh_i

                # The max value between the threshold and falling phase of an AP is the peak.
                peak_i = context_voltages[cur_trh_i:fall_trh_i].argmax() + cur_trh_i
                peak_mv = context_voltages[peak_i]
                amplitude = peak_mv - context_voltages[cur_trh_i]

                # Additionally check the peak voltage and amplitude.
                if peak_mv >= AP_MV_THRESHOLD and amplitude >= AP_MIN_AMPLITUDE:
                    peak_indexes.append(peak_i + start_index)
                    trh_indexes.append(cur_trh_i + start_index)

    return peak_indexes, trh_indexes
