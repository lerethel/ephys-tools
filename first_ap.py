import matplotlib.pyplot as plt
import numpy as np

import fn

AP_TIME_WINDOW = 0.002
FAHP_TIME_WINDOW = 0.005
MAHP_TIME_WINDOW = 0.050
HALF_WIDTH_POINTS_TO_INTERPOLATE = 100

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
AP_HALF_WIDTH_LINE_STYLE = {
    "color": "blue",
    "alpha": 0.7,
    "ls": "--",
    "label": "AP half-width",
}
AP_PEAK_LINE_STYLE = {
    "color": "orange",
    "alpha": 0.7,
    "ls": "--",
    "label": "AP peak",
}
AP_HALF_WIDTH_MARKER_STYLE = {
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

    def _get_ahp(self, start_index, end_index, second_ap_trh_i):
        if end_index >= self.step_end + 1:
            end_index = self.step_end

        # Check if there's an AP in the way.
        if second_ap_trh_i and end_index > second_ap_trh_i:
            return None

        ahp_context = self.abf.sweepY[start_index:end_index]
        ahp_index = ahp_context.argmin()

        # fAHP is searched for from the AP peak, so the parenthesized statement will return 0.
        # Other types of AHP aren't searched for from the AP peak, so we need to account for that.
        return ahp_index + (start_index - self.peak_i), ahp_context[ahp_index]

    def _get_half_width_params(self, ap_left_side):
        real_ap_half_ampl = self.props["amplitude"] / 2 + self.props["threshold"]

        ap_side_voltages = (
            self.abf.sweepY[self.prepeak_i : self.peak_i]
            if ap_left_side
            else self.abf.sweepY[self.peak_i : self.postpeak_i]
        )

        min_mv_index = (
            np.where(ap_side_voltages < real_ap_half_ampl)
            if ap_left_side
            else np.where(ap_side_voltages > real_ap_half_ampl)
        )[0][-1]

        interp_s = np.linspace(
            fn.sample_to_s(min_mv_index, self.abf),
            fn.sample_to_s(min_mv_index + 1, self.abf),
            HALF_WIDTH_POINTS_TO_INTERPOLATE,
        )

        interp_mV = np.linspace(
            ap_side_voltages[min_mv_index],
            ap_side_voltages[min_mv_index + 1],
            HALF_WIDTH_POINTS_TO_INTERPOLATE,
        )

        ap_half_width_info = fn.get_closest(interp_mV, real_ap_half_ampl)

        return interp_s[ap_half_width_info[0]], ap_half_width_info[1]

    def first_ap(self):
        abf = self.abf

        self.props = {}

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, channel=1)

            peak_indexes, trh_indexes = fn.find_aps(self.step_start, self.step_end, abf)

            try:
                peak_i = peak_indexes[0]
            except IndexError:
                continue

            try:
                second_ap_trh_i = trh_indexes[1]
            except IndexError:
                second_ap_trh_i = None

            abf.setSweep(sweep_no, channel=0)
            self.props["rheobase"] = fn.get_current_step(abf)

            abf.setSweep(sweep_no, channel=1)
            trh_i = trh_indexes[0]
            prepeak_i = peak_i - fn.s_to_sample(AP_TIME_WINDOW, abf)
            postpeak_i = peak_i + fn.s_to_sample(AP_TIME_WINDOW, abf)
            fahp_max_i = peak_i + fn.s_to_sample(FAHP_TIME_WINDOW, abf)
            mahp_max_i = peak_i + fn.s_to_sample(MAHP_TIME_WINDOW, abf)

            # Burst check
            if second_ap_trh_i and postpeak_i > second_ap_trh_i:
                postpeak_i = second_ap_trh_i

            self.peak_i = peak_i
            self.prepeak_i = prepeak_i
            self.postpeak_i = postpeak_i
            self.ahp_max_i = mahp_max_i

            self.props["latency"] = (
                fn.sample_to_s(
                    trh_i - self.step_start,
                    abf,
                )
                * 1000
                if self.props["rheobase"] != 0
                else 0.0
            )

            self.props["threshold"] = abf.sweepY[trh_i]

            # fAHP
            fahp_params = self._get_ahp(peak_i, fahp_max_i, second_ap_trh_i)

            if fahp_params:
                self.props["fAHP"] = fahp_params[1] - self.props["threshold"]

            # mAHP
            mahp_params = self._get_ahp(fahp_max_i, mahp_max_i, second_ap_trh_i)

            if mahp_params:
                self.props["mAHP"] = mahp_params[1] - self.props["threshold"]

            self.props["amplitude"] = abf.sweepY[peak_i] - self.props["threshold"]

            self.fahp_params = fahp_params
            self.mahp_params = mahp_params

            # Half-width
            self.half_width_left_params = self._get_half_width_params(True)
            self.half_width_right_params = self._get_half_width_params(False)

            self.props["half-width"] = (
                (self.half_width_right_params[0] + fn.sample_to_s(peak_i, abf))
                - (self.half_width_left_params[0] + fn.sample_to_s(prepeak_i, abf))
            ) * 1000

            # Max rise and decay slopes
            ap_derivative = fn.get_derivative(trh_i, postpeak_i, abf)
            self.props["max_rise_slope"] = ap_derivative.max()
            self.props["max_decay_slope"] = ap_derivative.min()

            return self.props

    def show_plot(self):
        abf = self.abf

        peak_i = self.peak_i
        prepeak_i = self.prepeak_i
        postpeak_i = self.postpeak_i
        postahp_i = self.ahp_max_i + fn.s_to_sample(0.002, abf)

        plt.figure(**fn.FIGURE_INIT_PARAMS)

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

        plt.subplot(223)
        plt.title(AP_UP_CLOSE_TITLE)
        plt.plot(
            abf.sweepX[prepeak_i:postpeak_i],
            abf.sweepY[prepeak_i:postpeak_i],
        )
        plt.axhline(self.props["threshold"], **AP_MV_MS_THRESHOLD_LINE_STYLE)
        plt.scatter(
            [
                fn.sample_to_s(prepeak_i, abf) + self.half_width_left_params[0],
                fn.sample_to_s(peak_i, abf) + self.half_width_right_params[0],
            ],
            [self.half_width_left_params[1], self.half_width_right_params[1]],
            **AP_HALF_WIDTH_MARKER_STYLE,
        )
        plt.axhline(self.half_width_left_params[1], **AP_HALF_WIDTH_LINE_STYLE)
        plt.axhline(abf.sweepY[peak_i], **AP_PEAK_LINE_STYLE)

        plt.subplot(224)
        plt.title(FAHP_MAHP_TITLE)
        plt.plot(
            abf.sweepX[prepeak_i:postahp_i],
            abf.sweepY[prepeak_i:postahp_i],
        )

        if self.fahp_params:
            plt.scatter(
                abf.sweepX[peak_i + self.fahp_params[0]],
                self.fahp_params[1],
                **AP_FAHP_MARKER_STYLE,
            )

        if self.mahp_params:
            plt.scatter(
                abf.sweepX[peak_i + self.mahp_params[0]],
                self.mahp_params[1],
                **AP_MAHP_MARKER_STYLE,
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
