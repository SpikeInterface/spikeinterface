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

def cross_correlate_with_scale(x, signa11_blanked, signal2_blanked, thr=100, plot=True):
    """
    """
    best_correlation = 0
    best_displacements = np.zeros_like(signa11_blanked)

    # TODO: use kriging interp

    xcorr = []

    for scale in np.linspace(0.85, 1.15, 10):

        nonzero = np.where(signa11_blanked > 0)[0]
        if not np.any(nonzero):
            continue

        midpoint = nonzero[0] + np.ptp(nonzero) / 2
        x_scale = (x - midpoint) * scale + midpoint

        interp_f = scipy.interpolate.interp1d(x_scale, signa11_blanked, fill_value=0.0, bounds_error=False)  # TODO: try cubic etc... or Kriging

        scaled_func = interp_f(x)

 #       plt.plot(signa11_blanked)
 #       plt.plot(scaled_func)
 #       plt.show()

       # breakpoint()

        for sh in np.arange(-thr, thr):  # TODO: we are off by one here

            shift_signal1_blanked = shift_array_fill_zeros(scaled_func, sh)

            x_shift = x_scale - sh  # TODO: rename

            # is this pull back?
         #   interp_f = scipy.interpolate.interp1d(xs, shift_signal1_blanked, fill_value=0.0, bounds_error=False)  # TODO: try cubic etc... or Kriging

          #  scaled_func = interp_f(x_shift)

            corr_value = np.correlate(
                    shift_signal1_blanked - np.mean(shift_signal1_blanked),
                    signal2_blanked - np.mean(signal2_blanked),
                ) / signa11_blanked.size

            if corr_value > best_correlation:
                best_displacements = x_shift
                best_correlation = corr_value

            if False and np.abs(sh) == 1:
                print(corr_value)

                plt.plot(shift_signal1_blanked)
                plt.plot(signal2_blanked)
                plt.show()
               # plt.draw()  # Draw the updated figure
               # plt.pause(0.1)  # Pause for 0.5 seconds before updating
               # plt.clf()

             #   breakpoint()


      #  xcorr.append(np.max(np.r_[xcorr_scale]))

    if False:
        xcorr = np.r_[xcorr]
    #    shift = np.argmax(xcorr) - thr

        print("MAX", np.max(xcorr))

        if np.max(xcorr) < 0.0001:
            shift = 0
        else:
            shift = np.argmax(xcorr) - thr

        print("output shift", shift)

    return best_displacements

# plt.plot(signal1)
# plt.plot(signal2)

def get_shifts(signal1, signal2, windows, plot=True):

    import matplotlib.pyplot as plt

    signa11_blanked = signal1.copy()
    signal2_blanked = signal2.copy()

    best_displacements = np.zeros_like(signal1)

    if (first_idx := windows[0][0]) != 0:
        print("first idx", first_idx)
        signa11_blanked[:first_idx] = 0
        signal2_blanked[:first_idx] = 0

    if (last_idx := windows[-1][-1]) != signal1.size - 1:  # double check
        print("last idx", last_idx)
        signa11_blanked[last_idx:] = 0
        signal2_blanked[last_idx:] = 0

    segment_shifts = np.empty(len(windows))


    x = np.arange(signa11_blanked.size)
    x_orig = x.copy()

    for round in range(len(windows)):

        #if round == 0:
        #    shift = cross_correlate(signa11_blanked, signal2_blanked, thr=100)  # for first rigid, do larger!
        #else:
        displacements = cross_correlate_with_scale(x, signa11_blanked, signal2_blanked, thr=200, plot=False)



      #  breakpoint()

        interpf = scipy.interpolate.interp1d(displacements, signa11_blanked, fill_value=0.0, bounds_error=False)  # TODO: move away from this indexing sceheme
        signa11_blanked = interpf(x)



  #      cum_shifts.append(shift)
 #       print("shift", shift)

        # shift the signal1, or use indexing

#        signa11_blanked = shift_array_fill_zeros(signa11_blanked, shift)  # INTERP HERE, KRIGING. but will accumulate interpolation errors...

    #    if plot:
     #       print("round", round)
      #      plt.plot(signa11_blanked)
       #     plt.plot(signal2_blanked)
        #    plt.show()

        window_corrs = np.empty(len(windows))
        for i, idx in enumerate(windows):
            window_corrs[i] = np.correlate(
                signa11_blanked[idx] - np.mean(signa11_blanked[idx]),
                signal2_blanked[idx] - np.mean(signal2_blanked[idx]),
            ) / signa11_blanked[idx].size

        max_window = np.argmax(window_corrs)  # TODO: cutoff!

        if False:
            small_shift = cross_correlate(signa11_blanked[windows[max_window]], signal2_blanked[windows[max_window]], thr=windows[max_window].size //2)
            signa11_blanked = shift_array_fill_zeros(signa11_blanked, small_shift)
            segment_shifts[max_window] = np.sum(cum_shifts) + small_shift

        best_displacements[windows[max_window]] = displacements[windows[max_window]]

        x = displacements

        signa11_blanked[windows[max_window]] = 0
        signal2_blanked[windows[max_window]] = 0

    # TODO: need to carry over displacements!

    print(best_displacements)
    interpf = scipy.interpolate.interp1d(best_displacements, signal1, fill_value=0.0, bounds_error=False)  # TODO: move away from this indexing sceheme
    final = interpf(x_orig)

    plt.plot(final)
    plt.plot(signal2)
    plt.show()

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
