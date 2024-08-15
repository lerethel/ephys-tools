import operator as op

import matplotlib.pyplot as plt
import numpy as np

import fn

AP_TIME_WINDOW = 0.002
FAHP_TIME_WINDOW = 0.005
MAHP_TIME_WINDOW = 0.050

RHEOBASE_TITLE = "Rheobase: %d pA. Analyzed AP is marked"
AP_UP_CLOSE_TITLE = "Analyzed AP up close"
FAHP_MAHP_TITLE = "fAHP and mAHP (missing means not found)"

STEP_BOUNDARIES_STYLE = {
    "color": "red",
    "alpha": 0.5,
    "ls": "--",
    "label": "Step boundaries",
}
AP_MV_MS_THRESHOLD_LINE_STYLE = {
    "color": "grey",
    "alpha": 0.7,
    "ls": "--",
    "label": f"AP threshold (> {fn.AP_MV_MS_RISE_THRESHOLD} mV/ms)",
}
AP_HALF_AMPLITUDE_LINE_STYLE = {
    "color": "blue",
    "alpha": 0.7,
    "ls": "--",
    "label": "AP half-amplitude",
}
AP_PEAK_LINE_STYLE = {
    "color": "orange",
    "alpha": 0.7,
    "ls": "--",
    "label": "AP peak",
}
AP_HALF_AMPLITUDE_MARKER_STYLE = {
    "color": "blue",
    "alpha": 0.7,
    "marker": ".",
}
AP_FAHP_MARKER_STYLE = {"color": "magenta", "label": "fAHP"}
AP_MAHP_MARKER_STYLE = {"color": "green", "label": "mAHP"}


class FirstAP:
    def __init__(self, abf):
        step_start, step_end = fn.get_step_boundaries(abf)

        self.abf = abf
        self.step_start = step_start
        self.step_end = step_end

    def _get_ahp_info(self, start_index, end_index, second_ap_trh_i):
        end_index = min(end_index, self.step_end)

        # Check if there's an AP in the way.
        if second_ap_trh_i and end_index > second_ap_trh_i:
            return None

        ahp_context = self.abf.sweepY[start_index:end_index]
        ahp_index = ahp_context.argmin()

        return ahp_index + start_index, ahp_context[ahp_index]

    def _interpolate_side(self, start_index, end_index, operation):
        abf = self.abf
        half_amplitude = self.props["amplitude"] / 2 + self.props["threshold"]
        side_voltages = abf.sweepY[start_index:end_index]
        points_to_interpolate = int(1e6 / abf.sampleRate)

        # When operation is lt: Go up the left side of an AP and find the last value lower than
        # the calculated half-amplitude. When operation is gt: Go down the right side of the AP
        # and find the last value higher than the calculated half-amplitude.
        closest_voltage_i = np.where(operation(side_voltages, half_amplitude))[0][-1]

        interp_time = np.linspace(
            fn.sample_to_s(closest_voltage_i, abf),
            fn.sample_to_s(closest_voltage_i + 1, abf),
            points_to_interpolate,
        )

        interp_voltage = np.linspace(
            side_voltages[closest_voltage_i],
            side_voltages[closest_voltage_i + 1],
            points_to_interpolate,
        )

        half_amplitude_info = fn.get_closest(interp_voltage, half_amplitude)

        return (
            interp_time[half_amplitude_info[0]] + fn.sample_to_s(start_index, abf),
            half_amplitude_info[1],
        )

    def _get_half_amplitude_info(self):
        left_side_info = self._interpolate_side(self.trh_i, self.peak_i, op.lt)
        right_side_info = self._interpolate_side(self.peak_i, self.postpeak_i, op.gt)

        return {"left": left_side_info, "right": right_side_info}

    def first_ap(self):
        abf = self.abf

        self.props = {}

        for sweep_i in abf.sweepList:
            abf.setSweep(sweep_i, channel=fn.CURRENT_CLAMP_CHANNEL)

            peak_indexes, trh_indexes = fn.find_aps(self.step_start, self.step_end, abf)

            try:
                peak_i = peak_indexes[0]
            except IndexError:
                continue

            try:
                second_ap_trh_i = trh_indexes[1]
            except IndexError:
                second_ap_trh_i = None

            trh_i = trh_indexes[0]
            prepeak_i = peak_i - fn.s_to_sample(AP_TIME_WINDOW, abf)
            postpeak_i = peak_i + fn.s_to_sample(AP_TIME_WINDOW, abf)
            fahp_max_i = peak_i + fn.s_to_sample(FAHP_TIME_WINDOW, abf)
            mahp_max_i = peak_i + fn.s_to_sample(MAHP_TIME_WINDOW, abf)

            # Make sure the window doesn't include any other APs in case of a burst.
            if second_ap_trh_i and postpeak_i > second_ap_trh_i:
                postpeak_i = second_ap_trh_i

            self.peak_i = peak_i
            self.trh_i = trh_i
            self.prepeak_i = prepeak_i
            self.postpeak_i = postpeak_i
            self.ahp_max_i = mahp_max_i

            self.props["rheobase"] = fn.get_current_step(abf)
            self.props["latency"] = (
                fn.sample_to_s(trh_i - self.step_start, abf) * 1000
                if self.props["rheobase"] != 0
                else 0.0
            )

            self.props["threshold"] = abf.sweepY[trh_i]
            self.props["amplitude"] = abf.sweepY[peak_i] - self.props["threshold"]

            half_amplitude_info = self._get_half_amplitude_info()
            self.half_amplitude_left_info = half_amplitude_info["left"]
            self.half_amplitude_right_info = half_amplitude_info["right"]
            self.props["half-width"] = (
                self.half_amplitude_right_info[0] - self.half_amplitude_left_info[0]
            ) * 1000

            ap_derivative = fn.get_derivative(trh_i, postpeak_i, abf)
            self.props["max_rise_slope"] = ap_derivative.max()
            self.props["max_decay_slope"] = ap_derivative.min()

            # fAHP
            fahp_info = self._get_ahp_info(peak_i, fahp_max_i, second_ap_trh_i)

            if fahp_info:
                self.props["fAHP"] = fahp_info[1] - self.props["threshold"]

            # mAHP
            mahp_info = self._get_ahp_info(fahp_max_i, mahp_max_i, second_ap_trh_i)

            if mahp_info:
                self.props["mAHP"] = mahp_info[1] - self.props["threshold"]

            self.fahp_info = fahp_info
            self.mahp_info = mahp_info

            return self.props

    def show_plot(self):
        abf = self.abf

        peak_i = self.peak_i
        prepeak_i = self.prepeak_i
        postpeak_i = self.postpeak_i
        postahp_i = self.ahp_max_i + fn.s_to_sample(0.002, abf)

        plt.figure(**fn.FIGURE_INIT_PARAMS)

        # The whole step with the analyzed AP marked.
        plt.subplot(2, 2, (1, 2))
        plt.title(RHEOBASE_TITLE % self.props["rheobase"])
        plt.plot(*fn.extend_coords(self.step_start, 0.05, self.step_end, 0.2, abf))
        plt.scatter(
            abf.sweepX[peak_i],
            abf.sweepY[peak_i] + fn.DISTANCE_BETWEEN_MARKERS_AND_MAX_PEAK,
            **fn.AP_MARKER_STYLE,
        )

        if self.props["rheobase"] == 0:
            plt.axvline(abf.sweepX[self.step_start], **STEP_BOUNDARIES_STYLE)
            plt.axvline(abf.sweepX[self.step_end], **STEP_BOUNDARIES_STYLE)

        plt.axhline(self.props["threshold"], **AP_MV_MS_THRESHOLD_LINE_STYLE)

        # Only the analyzed AP up close.
        plt.subplot(223)
        plt.title(AP_UP_CLOSE_TITLE)
        plt.plot(abf.sweepX[prepeak_i:postpeak_i], abf.sweepY[prepeak_i:postpeak_i])
        plt.axhline(self.props["threshold"], **AP_MV_MS_THRESHOLD_LINE_STYLE)
        plt.scatter(
            [self.half_amplitude_left_info[0], self.half_amplitude_right_info[0]],
            [self.half_amplitude_left_info[1], self.half_amplitude_right_info[1]],
            **AP_HALF_AMPLITUDE_MARKER_STYLE,
        )
        plt.axhline(self.half_amplitude_left_info[1], **AP_HALF_AMPLITUDE_LINE_STYLE)
        plt.axhline(abf.sweepY[peak_i], **AP_PEAK_LINE_STYLE)

        # The analyzed AP with part of the step after it to see AHPs.
        plt.subplot(224)
        plt.title(FAHP_MAHP_TITLE)
        plt.plot(abf.sweepX[prepeak_i:postahp_i], abf.sweepY[prepeak_i:postahp_i])

        if self.fahp_info:
            plt.scatter(
                abf.sweepX[self.fahp_info[0]], self.fahp_info[1], **AP_FAHP_MARKER_STYLE
            )

        if self.mahp_info:
            plt.scatter(
                abf.sweepX[self.mahp_info[0]], self.mahp_info[1], **AP_MAHP_MARKER_STYLE
            )

        fn.set_all_xylabels("Time (s)", "Voltage (mV)")
        fn.show_plot(abf)

    @classmethod
    def use(cls, abf, show_plot=True):
        inst = cls(abf)
        props = inst.first_ap()

        if show_plot:
            inst.show_plot()

        return props


def first_ap(abf, show_plot=True):
    return FirstAP.use(abf, show_plot)


if __name__ == "__main__":
    import pyabf

    print(first_ap(pyabf.ABF("abf/aps/aps_01.abf")))
