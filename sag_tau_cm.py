import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import fn

TAU_DEFAULT_CURRENT_STEP = -200
SAG_DEFAULT_CURRENT_STEP = -200

PEAK_TIME_WINDOW = 0.15
SS_TIME_WINDOW = 0.05

EXTRAPOLATED_RESPONSE_TITLE = "Extrapolation from tau (%d pA)"
TAU_TITLE = "Tau (%d pA)"
SAG_TITLE = "Sag (%d pA)"

PERCENT_OF_PEAK_LINE_STYLE = {
    "color": "grey",
    "alpha": 0.7,
    "ls": "--",
    "label": r"10-95% of peak",
}
EXTRAPOLATED_RESPONSE_STYLE = {
    "color": "magenta",
    "alpha": 0.7,
    "ls": "--",
    "label": "Extrapolated response",
}
PEAK_AND_SS_MARKER_STYLE = {
    "color": "purple",
    "label": "Sag peak and steady state",
}


class Sag:
    def __init__(self, abf):
        step_start, step_end = fn.get_step_boundaries(abf)

        self.abf = abf
        self.step_start = step_start
        self.step_end = step_end
        self.start_offset = step_start + fn.s_to_sample(PEAK_TIME_WINDOW, abf)
        self.end_offset = step_end - fn.s_to_sample(SS_TIME_WINDOW, abf)

    def _double_exp(self, t, a, tau_1, b, tau_2, c):
        return a * np.exp(-t / tau_1) + b * np.exp(-t / tau_2) + c

    def _get_peak_pct_info(self, percent, props):
        index, value = fn.get_closest(
            props["peak_window"],
            (props["peak_from_bsl"] * (percent / 100)) + props["bsl"],
        )
        return index + self.step_start, value

    def _get_step_props(self, current_step):
        abf = self.abf

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, channel=1)

            if fn.get_current_step(abf) == current_step:
                peak_search_window = abf.sweepY[self.step_start : self.start_offset]
                ss_window = abf.sweepY[self.end_offset : self.step_end]

                baseline = peak_search_window.max()
                peak_i = peak_search_window.argmin()
                ss_i = ss_window.argmin()
                peak = peak_search_window[peak_i]
                ss = ss_window[ss_i]

                return {
                    # Include only the peak without the rising phase.
                    "peak_window": abf.sweepY[
                        self.step_start : self.step_start + peak_i + 1
                    ],
                    "ss_window": ss_window,
                    "bsl": baseline,
                    "peak": peak,
                    "ss": ss,
                    "peak_i": peak_i + self.step_start,
                    "ss_i": ss_i + self.end_offset,
                    "peak_from_bsl": peak - baseline,
                    "ss_from_bsl": ss - baseline,
                    "sweep_no": sweep_no,
                    "current_step": current_step,
                }

    def _get_best_fit_props(self, tau_props, peak_10_pct_i, peak_95_pct_i):
        abf = self.abf

        x_fit = (
            abf.sweepX[peak_10_pct_i:peak_95_pct_i] - abf.sweepX[peak_10_pct_i]
        ) * 1000
        y_fit = abf.sweepY[peak_10_pct_i:peak_95_pct_i]

        fits = {"popts": [], "r_squared": []}

        init_c = abf.sweepY[peak_95_pct_i] - 5

        # Iterate over a range of possible tau values to find the one with the highest r-squared.
        # If a value doesn't fit at all, curve_fit will throw a RuntimeError, which is ignored.
        for init_tau_1 in range(30, 0, -10):
            init_a = init_tau_1

            try:
                cur_popt = curve_fit(
                    self._double_exp,
                    x_fit,
                    y_fit,
                    p0=(
                        init_a,
                        init_tau_1,
                        init_a / 10,
                        init_tau_1 / 10,
                        init_c,
                    ),
                )[0]
            except RuntimeError:
                pass
            else:
                # RSS: residual sum of squares
                rss = np.sum(np.square(y_fit - self._double_exp(x_fit, *cur_popt)))
                # TSS: total sum of squares
                tss = np.sum(np.square(y_fit - np.mean(y_fit)))

                fits["r_squared"].append(1 - rss / tss)
                fits["popts"].append(cur_popt)

        best_fit_index = fits["r_squared"].index(max(fits["r_squared"]))
        best_popt = fits["popts"][best_fit_index]

        x_extrap = (abf.sweepX[peak_10_pct_i:] - abf.sweepX[peak_10_pct_i]) * 1000
        y_extrap = self._double_exp(x_extrap, *best_popt)

        rm_extrap = (y_extrap[-1] - tau_props["bsl"]) / tau_props["current_step"]
        tau = max(best_popt[1], best_popt[3])
        cm = tau / rm_extrap

        return tau, cm, y_extrap

    def tau_cm(self, current_step=TAU_DEFAULT_CURRENT_STEP):
        tau_props = self._get_step_props(current_step)
        peak_10_pct_i = self._get_peak_pct_info(10, tau_props)[0]
        peak_95_pct_i = self._get_peak_pct_info(95, tau_props)[0]

        tau, cm, y_extrap = self._get_best_fit_props(
            tau_props, peak_10_pct_i, peak_95_pct_i
        )

        self.tau_props = tau_props
        self.y_extrap = y_extrap
        self.peak_10_pct_i = peak_10_pct_i
        self.peak_95_pct_i = peak_95_pct_i

        return {
            "tau": tau,
            "Cm": cm,
            "current_step": tau_props["current_step"],
        }

    def sag_ratio(self, current_step=SAG_DEFAULT_CURRENT_STEP):
        sag_props = self._get_step_props(current_step)
        sag_ratio = (sag_props["peak_from_bsl"] - sag_props["ss_from_bsl"]) / sag_props[
            "peak_from_bsl"
        ]

        self.sag_props = sag_props

        return {
            "sag_ratio": sag_ratio if sag_ratio > 0 else 0,
            "current_step": sag_props["current_step"],
        }

    def show_plot(self):
        abf = self.abf
        tau_props = self.tau_props
        sag_props = self.sag_props

        plt.figure(**fn.FIGURE_INIT_PARAMS)

        abf.setSweep(tau_props["sweep_no"], channel=1)

        plt.subplot(221)
        plt.title(EXTRAPOLATED_RESPONSE_TITLE % tau_props["current_step"])
        plt.plot(abf.sweepX, abf.sweepY)
        plt.plot(
            abf.sweepX[self.peak_10_pct_i :],
            self.y_extrap,
            **EXTRAPOLATED_RESPONSE_STYLE,
        )

        plt.subplot(222)
        plt.title(TAU_TITLE % tau_props["current_step"])
        plt.plot(*fn.extend_coords(self.step_start, 0.01, self.start_offset, 0.2, abf))
        plt.axvline(
            fn.sample_to_s(self.peak_10_pct_i, abf),
            **PERCENT_OF_PEAK_LINE_STYLE,
        )
        plt.axvline(
            fn.sample_to_s(self.peak_95_pct_i, abf),
            **PERCENT_OF_PEAK_LINE_STYLE,
        )

        abf.setSweep(sag_props["sweep_no"], channel=1)

        plt.subplot(2, 2, (3, 4))
        plt.title(SAG_TITLE % sag_props["current_step"])
        plt.plot(*fn.extend_coords(self.step_start, 0.05, self.step_end, 0.2, abf))
        plt.scatter(
            fn.sample_to_s(sag_props["peak_i"], abf),
            sag_props["peak"],
            **PEAK_AND_SS_MARKER_STYLE,
        )
        plt.scatter(
            fn.sample_to_s(sag_props["ss_i"], abf),
            sag_props["ss"],
            **PEAK_AND_SS_MARKER_STYLE,
        )

        fn.set_all_xylabels("Time (s)", "Voltage (mV)")
        fn.show_plot(abf)

    @classmethod
    def use(cls, abf, show_plot=True):
        inst = cls(abf)
        tau_cm = inst.tau_cm()
        sag_ratio = inst.sag_ratio()

        if show_plot:
            inst.show_plot()

        return {"tau": tau_cm, "sag": sag_ratio}


def sag_tau_cm(abf, show_plot=True):
    return Sag.use(abf, show_plot)


if __name__ == "__main__":
    import pyabf

    print(sag_tau_cm(pyabf.ABF("abf/sag/sag_01.abf")))
