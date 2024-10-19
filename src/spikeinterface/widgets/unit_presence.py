from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class UnitPresenceWidget(BaseWidget):
    """
    Estimates of the probability density function for each unit using Gaussian kernels,

    Parameters
    ----------
    sorting : SortingExtractor
        The sorting extractor object
    segment_index : None or int
        The segment index.
    time_range : list or None, default: None
        List with start time and end time
    bin_duration_s : float, default: 0.5
        Bin size (in seconds) for the heat map time axis
    smooth_sigma : float, default: 4.5
        Sigma for the Gaussian kernel (in number of bins)
    """

    def __init__(
        self,
        sorting,
        segment_index=None,
        time_range=None,
        bin_duration_s=0.05,
        smooth_sigma=4.5,
        backend=None,
        **backend_kwargs,
    ):
        sorting = self.ensure_sorting(sorting)

        if segment_index is None:
            nseg = sorting.get_num_segments()
            if nseg != 1:
                raise ValueError("You must provide segment_index=...")
            else:
                segment_index = 0

        data_plot = dict(
            sorting=sorting,
            segment_index=segment_index,
            time_range=time_range,
            bin_duration_s=bin_duration_s,
            smooth_sigma=smooth_sigma,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        # backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        # self.make_mpl_figure(**backend_kwargs)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        sorting = dp.sorting

        spikes = sorting.to_spike_vector(concatenated=False, use_cache=True)
        spikes = spikes[dp.segment_index]

        fs = sorting.get_sampling_frequency()

        if dp.time_range is not None:
            t0, t1 = dp.time_range
            ind0 = int(t0 * fs)
            ind1 = int(t1 * fs)
            mask = (spikes["sample_index"] >= ind0) & (spikes["sample_index"] <= ind1)
            spikes = spikes[mask]

        if spikes.size == 0:
            return

        last = spikes["sample_index"][-1]
        max_time = last / fs

        num_units = len(sorting.unit_ids)
        num_time_bins = int(max_time / dp.bin_duration_s) + 1
        map = np.zeros((num_units, num_time_bins))
        ind0 = spikes["unit_index"]
        ind1 = spikes["sample_index"] // int(dp.bin_duration_s * fs)
        map[ind0, ind1] += 1

        if dp.smooth_sigma is not None:
            import scipy.signal

            n = int(dp.smooth_sigma * 5)
            bins = np.arange(-n, n + 1)
            smooth_kernel = np.exp(-(bins**2) / (2 * dp.smooth_sigma**2))
            smooth_kernel /= np.sum(smooth_kernel)
            smooth_kernel = smooth_kernel[np.newaxis, :]
            map = scipy.signal.oaconvolve(map, smooth_kernel, mode="same", axes=1)

        im = self.ax.matshow(map, cmap="inferno", aspect="auto")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Units")

        self.figure.colorbar(im)
