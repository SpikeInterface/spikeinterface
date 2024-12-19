import numpy as np
import matplotlib.pyplot as plt
import scipy


def shift_array_fill_zeros(array: np.ndarray, shift: int) -> np.ndarray:
    abs_shift = np.abs(shift)
    pad_tuple = (0, abs_shift) if shift > 0 else (abs_shift, 0)
    padded_hist = np.pad(array, pad_tuple, mode="constant")
    cut_padded_array = padded_hist[abs_shift:] if shift >= 0 else padded_hist[:-abs_shift]
    return cut_padded_array


# Load and normalize signals
signal1 = np.load(r"C:\Users\Joe\work\git-repos\forks\spikeinterface\debugging\signal1_1.npy")
signal2 = np.load(r"C:\Users\Joe\work\git-repos\forks\spikeinterface\debugging\signal2_1.npy")


def cross_correlate(sig1, sig2, thr= None):
    xcorr = np.correlate(sig1, sig2, mode="full")

    n = sig1.size
    low_cut_idx = np.arange(0, n - thr)  # double check
    high_cut_idx = np.arange(n + thr, 2 * n - 1)

    xcorr[low_cut_idx] = 0
    xcorr[high_cut_idx] = 0

    if np.max(xcorr) < 0.01:
        shift = 0
    else:
        shift = np.argmax(xcorr) - xcorr.size // 2

    return shift

def cross_correlate_with_scale(signa11_blanked, signal2_blanked, thr=100, plot=True):
    """
    """
    xcorr = []
    for s in np.arange(-thr, thr):  # TODO: we are off by one here

        shift_signal1_blanked = shift_array_fill_zeros(signa11_blanked, s)

        x = np.arange(shift_signal1_blanked.size)

        xcorr_scale = []
        for scale in np.linspace(0.75, 1.25, 10):

            midpoint = np.argmax(shift_signal1_blanked)  # assumes x is 0 .. n TODO: IMPROVE
            xs = (x - midpoint) * scale + midpoint

            # is this pull back?
            interp_f = scipy.interpolate.interp1d(xs, shift_signal1_blanked, fill_value=0.0, bounds_error=False)  # TODO: try cubic etc... or Kriging

            scaled_func = interp_f(x)

            corr_value = np.correlate(
                    scaled_func - np.mean(scaled_func),
                    signal2_blanked - np.mean(signal2_blanked),
                ) / signa11_blanked.size

            xcorr_scale.append(
                corr_value
            )

            if plot and corr_value > 0.0045: # and np.abs(s) < 10:
                print(corr_value)

                plt.plot(shift_signal1_blanked)
                plt.plot(signal2_blanked)
                plt.show()

                plt.plot(scaled_func)
                plt.plot(signal2_blanked)
                plt.show()
          #      plt.title(f"corr value: {corr_value}")
           #     plt.draw()  # Draw the updated figure
            #    plt.pause(0.1)  # Pause for 0.5 seconds before updating
             #   plt.clf()

        xcorr.append(np.max(np.r_[xcorr_scale]))

    xcorr = np.r_[xcorr]
#    shift = np.argmax(xcorr) - thr

    print("MAX", np.max(xcorr))

    if np.max(xcorr) < 0.0001:
        shift = 0
    else:
        shift = np.argmax(xcorr) - thr

    print("output shift", shift)

    return shift

# plt.plot(signal1)
# plt.plot(signal2)

def get_shifts(signal1, signal2, windows, plot=True):

    import matplotlib.pyplot as plt

    signa11_blanked = signal1.copy()
    signal2_blanked = signal2.copy()

    if (first_idx := windows[0][0]) != 0:
        print("first idx", first_idx)
        signa11_blanked[:first_idx] = 0
        signal2_blanked[:first_idx] = 0

    if (last_idx := windows[-1][-1]) != signal1.size - 1:  # double check
        print("last idx", last_idx)
        signa11_blanked[last_idx:] = 0
        signal2_blanked[last_idx:] = 0

    segment_shifts = np.empty(len(windows))
    cum_shifts = []


    for round in range(len(windows)):

        if round == 0:
            shift = cross_correlate(signa11_blanked, signal2_blanked, thr=100)  # for first rigid, do larger!
        else:
            shift = cross_correlate_with_scale(signa11_blanked, signal2_blanked, thr=100, plot=False)


        cum_shifts.append(shift)
        print("shift", shift)

        # shift the signal1, or use indexing

        signa11_blanked = shift_array_fill_zeros(signa11_blanked, shift)

        if plot:
            print("round", round)
            plt.plot(signa11_blanked)
            plt.plot(signal2_blanked)
            plt.show()

        window_corrs = np.empty(len(windows))
        for i, idx in enumerate(windows):
            window_corrs[i] = np.correlate(
                signa11_blanked[idx] - np.mean(signa11_blanked[idx]),
                signal2_blanked[idx] - np.mean(signal2_blanked[idx]),
            ) / signa11_blanked[idx].size

        max_window = np.argmax(window_corrs)  # TODO: cutoff!

        small_shift = cross_correlate(signa11_blanked[windows[max_window]], signal2_blanked[windows[max_window]], thr=windows[max_window].size //2)

        signa11_blanked = shift_array_fill_zeros(signa11_blanked, small_shift)

        segment_shifts[max_window] = np.sum(cum_shifts) + small_shift

        signa11_blanked[windows[max_window]] = 0
        signal2_blanked[windows[max_window]] = 0

    return segment_shifts


num_windows = 5

windows = np.arange(signal1.size)

windows = np.array_split(windows, num_windows)

shifts = get_shifts(signal1, signal2, windows)

if False:

    shifts[0::2] = np.array(shifts1)  # TODO: MOVE
    shifts[1::2] = np.array(shifts2)

    breakpoint()
    print("done")
