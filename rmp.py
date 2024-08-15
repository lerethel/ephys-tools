import numpy as np

import fn


def rmp(abf, show_plot=True):
    for sweep_i in abf.sweepList:
        abf.setSweep(sweep_i, channel=fn.CURRENT_CLAMP_CHANNEL)

        if fn.get_current_step(abf) == 0:
            return np.mean(abf.sweepY)
    return None


if __name__ == "__main__":
    import pyabf

    print(rmp(pyabf.ABF("abf/aps/aps_01.abf")))
