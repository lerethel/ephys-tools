import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import fn

ANALYSIS_START_CURRENT_STEP = 0
ANALYSIS_END_CURRENT_STEP = 200
ADAPTATION_MIN_ISI_NUMBER = 5

GENERAL_STEP_TITLE = "Step: %d pA. APs found: %d"
RHEOBASE_STEP_TITLE = "Step: %d pA (rheobase). APs found: %d"
ONLY_RHEOBASE_SWEEP_MSG = "Only the rheobase sweep could be analyzed"

AP_MV_THRESHOLD_LINE_STYLE = {
    "color": "green",
    "alpha": 0.7,
    "ls": "--",
    "label": f"AP threshold ({fn.AP_MV_THRESHOLD} mV)",
}


class Firing:
    adaptation_analysis_current = 200

    def __init__(self, abf):
        step_start, step_end = fn.get_step_boundaries(abf)

        self.abf = abf
        self.step_start = step_start
        self.step_end = step_end

    def _get_freq_props(self, peak_indexes):
        abf = self.abf

        isis = []
        inst_freqs = []
        prev_i = None

        for cur_i in peak_indexes:
            if prev_i:
                cur_isi = fn.sample_to_s(cur_i, abf) - fn.sample_to_s(prev_i, abf)
                isis.append(cur_isi * 1000)
                inst_freqs.append(1 / cur_isi)
            prev_i = cur_i

        return isis, inst_freqs

    def _fill_subplot(self, subplot_index, info_index):
        abf = self.abf
        subplot = self.subplots[subplot_index]
        sweep_no, current_step, peak_indexes = self.step_info[info_index]

        title = (RHEOBASE_STEP_TITLE if info_index == 0 else GENERAL_STEP_TITLE) % (
            current_step,
            len(peak_indexes),
        )

        abf.setSweep(sweep_no, channel=1)
        subplot.set_title(title)
        subplot.plot(*fn.extend_coords(self.step_start, 0.02, self.step_end, 0.1, abf))
        subplot.scatter(
            self.abf.sweepX.take(peak_indexes),
            [
                self.abf.sweepY.take(peak_indexes).max()
                + fn.DISTANCE_BETWEEN_MARKERS_AND_MAX_PEAK
            ]
            * len(peak_indexes),
            **fn.AP_MARKER_STYLE,
        )
        subplot.axhline(fn.AP_MV_THRESHOLD, **AP_MV_THRESHOLD_LINE_STYLE)
        subplot.set_xlabel("Time (s)")
        subplot.set_ylabel("Voltage (mV)")

    def firing(self):
        abf = self.abf

        self.step_info = []

        props = {"current_steps": [], "AP_numbers": [], "mean_inst_freqs": []}
        rheobase_current = None

        for sweep_no in abf.sweepList:
            abf.setSweep(sweep_no, channel=1)

            current_step = fn.get_current_step(abf)
            peak_indexes = fn.find_aps(self.step_start, self.step_end, abf)[0]
            step_info = (sweep_no, current_step, peak_indexes)
            inst_freqs = None

            if rheobase_current is None and len(peak_indexes):
                rheobase_current = current_step
                self.step_info.append(step_info)

            if current_step == self.adaptation_analysis_current:
                isis, inst_freqs = self._get_freq_props(peak_indexes)

                if len(isis) >= ADAPTATION_MIN_ISI_NUMBER:
                    props["adaptation"] = {
                        "current_step": current_step,
                        "ISIs": isis,
                        "inst_freqs": inst_freqs,
                    }

            if ANALYSIS_START_CURRENT_STEP <= current_step <= ANALYSIS_END_CURRENT_STEP:
                if rheobase_current is not None and current_step != rheobase_current:
                    self.step_info.append(step_info)

                if not inst_freqs:
                    inst_freqs = self._get_freq_props(peak_indexes)[1]

                props["current_steps"].append(current_step)
                props["AP_numbers"].append(len(peak_indexes))
                props["mean_inst_freqs"].append(
                    np.mean(inst_freqs) if len(inst_freqs) else 0
                )

        return props

    def show_plot(self):
        self.subplots = []

        plt.figure(**fn.FIGURE_INIT_PARAMS)

        self.subplots.append(plt.subplot(211))
        self._fill_subplot(0, 0)

        self.subplots.append(plt.subplot(212))
        try:
            self._fill_subplot(1, 1)
        except IndexError:
            self.subplots[1].set_axis_off()
            self.subplots[1].text(
                0.5,
                0.5,
                ONLY_RHEOBASE_SWEEP_MSG,
                ha="center",
            )
        else:
            # Buttons; show only if there's more than one sweep.
            switcher = _SweepSwitcher(self)

            axswitch_next = plt.axes([0.80, 0.01, 0.12, 0.05])
            bswitch_next = Button(axswitch_next, "Next pair")
            bswitch_next.on_clicked(switcher.next_pair)

            axswitch_prev = plt.axes([0.65, 0.01, 0.12, 0.05])
            bswitch_prev = Button(axswitch_prev, "Previous pair")
            bswitch_prev.on_clicked(switcher.prev_pair)

        fn.show_plot(self.abf)

    @classmethod
    def use(cls, abf, show_plot=True):
        inst = cls(abf)
        props = inst.firing()

        if show_plot:
            inst.show_plot()

        return props


class _SweepSwitcher:
    def __init__(self, firing_self):
        self.firing = firing_self
        self.index = 0

    def _switch_pair(self):
        for i, subplot in enumerate(self.firing.subplots):
            subplot.clear()
            self.firing._fill_subplot(i, self.index + i)
        plt.draw()

    def next_pair(self, event):
        self.index += 2
        step_number = len(self.firing.step_info)

        if self.index == step_number - 1:
            self.index -= 1
        elif self.index >= step_number:
            self.index = 0

        self._switch_pair()

    def prev_pair(self, event):
        self.index -= 2

        if self.index == -1:
            self.index += 1
        elif self.index < 0:
            self.index = len(self.firing.step_info) - 2

        self._switch_pair()


def firing(abf, show_plot=True):
    return Firing.use(abf, show_plot)


if __name__ == "__main__":
    import pyabf

    print(firing(pyabf.ABF("abf/aps/aps_01.abf")))
