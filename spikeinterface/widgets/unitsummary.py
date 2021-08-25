import numpy as np
from matplotlib import pyplot as plt
from .basewidget import BaseWidget

from .utils import get_unit_colors

from .unitprobemap import plot_unit_probe_map
from .unitwaveformdensitymap import plot_unit_waveform_density_map
from .amplitudes import plot_amplitudes_timeseries
from .unitwaveforms import plot_unit_waveforms
from .isidistribution import plot_isi_distribution


class UnitSummaryWidget(BaseWidget):
    """
    Plot a unit summary.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    unit_id: into or str
        The unit id to plot the summary of
    amplitudes: dict or None
        Amplitudes 'by_unit' as returned by the st.postprocessing.get_spike_amplitudes(..., output="by_unit") function
    unit_colors: list or None
        Optional matplotlib color for the unit
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: UnitSummaryWidget
        The output widget
    """

    def __init__(self, waveform_extractor, unit_id, amplitudes=None,
                 unit_colors=None, figure=None, ax=None):

        assert ax is None
        # ~ assert axes is None

        if figure is None:
            figure = plt.figure(constrained_layout=False, figsize=(15, 7), )

        BaseWidget.__init__(self, figure, None)

        self.waveform_extractor = waveform_extractor
        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting
        self.unit_id = unit_id

        if unit_colors is None:
            unit_colors = get_unit_colors(self.sorting)
        self.unit_colors = unit_colors

        self.amplitudes = amplitudes

    def plot(self):
        we = self.waveform_extractor

        fig = self.figure
        self.ax.remove()

        if self.amplitudes is None:
            nrows = 2
        else:
            nrows = 3

        gs = fig.add_gridspec(nrows, 6)

        ax = fig.add_subplot(gs[:, 0])
        plot_unit_probe_map(we, unit_ids=[self.unit_id], axes=[ax], colorbar=False)
        ax.set_title('')

        ax = fig.add_subplot(gs[0:2, 1:3])
        plot_unit_waveforms(we, unit_ids=[self.unit_id], radius_um=60, axes=[ax], unit_colors=self.unit_colors)

        ax = fig.add_subplot(gs[0:2, 3:5])
        plot_unit_waveform_density_map(we, unit_ids=[self.unit_id], max_channels=1, ax=ax, same_axis=True)

        ax = fig.add_subplot(gs[0:2, 5])
        plot_isi_distribution(we.sorting, unit_ids=[self.unit_id], axes=[ax])
        ax.set_title('')

        if self.amplitudes is not None:
            ax = fig.add_subplot(gs[-1, 1:])
            plot_amplitudes_timeseries(we, amplitudes=self.amplitudes, unit_ids=[self.unit_id], ax=ax,
                                       unit_colors=self.unit_colors)
        ax.set_title('')

        fig.suptitle(f'Unit ID: {self.unit_id}')


def plot_unit_summary(*args, **kwargs):
    W = UnitSummaryWidget(*args, **kwargs)
    W.plot()
    return W


plot_unit_summary.__doc__ = UnitSummaryWidget.__doc__
