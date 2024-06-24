from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr, default_backend_kwargs


class RasterWidget(BaseWidget):
    """
    Plots spike train rasters.

    Parameters
    ----------
    sorting : SortingExtractor
        The sorting extractor object
    segment_index : None or int
        The segment index.
    unit_ids : list
        List of unit ids
    time_range : list
        List with start time and end time
    color : matplotlib color
        The color to be used
    """

    def __init__(
        self, sorting, segment_index=None, unit_ids=None, time_range=None, color="k", backend=None, **backend_kwargs
    ):
        sorting = self.ensure_sorting(sorting)

        if segment_index is None:
            if sorting.get_num_segments() != 1:
                raise ValueError("You must provide segment_index=...")
            segment_index = 0

        if time_range is None:
            frame_range = [0, sorting.to_spike_vector()[-1]["sample_index"]]
            time_range = [f / sorting.sampling_frequency for f in frame_range]
        else:
            assert len(time_range) == 2, "'time_range' should be a list with start and end time in seconds"
            frame_range = [int(t * sorting.sampling_frequency) for t in time_range]

        plot_data = dict(
            sorting=sorting,
            segment_index=segment_index,
            unit_ids=unit_ids,
            color=color,
            frame_range=frame_range,
            time_range=time_range,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        sorting = dp.sorting

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        units_ids = dp.unit_ids
        if units_ids is None:
            units_ids = sorting.unit_ids

        with plt.rc_context({"axes.edgecolor": "gray"}):
            for unit_index, unit_id in enumerate(units_ids):
                spiketrain = sorting.get_unit_spike_train(
                    unit_id,
                    start_frame=dp.frame_range[0],
                    end_frame=dp.frame_range[1],
                    segment_index=dp.segment_index,
                )
                spiketimes = spiketrain / float(sorting.sampling_frequency)
                self.ax.plot(
                    spiketimes,
                    unit_index * np.ones_like(spiketimes),
                    marker="|",
                    mew=1,
                    markersize=3,
                    ls="",
                    color=dp.color,
                )
            self.ax.set_yticks(np.arange(len(units_ids)))
            self.ax.set_yticklabels(units_ids)
            self.ax.set_xlim(*dp.time_range)
            self.ax.set_xlabel("time (s)")
