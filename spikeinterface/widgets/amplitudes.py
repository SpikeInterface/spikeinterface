import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget

from ..toolkit import get_spike_amplitudes
from .utils import get_unit_colors


class AmplitudeBaseWidget(BaseWidget):
    def __init__(self, waveform_extractor, unit_ids=None, amplitudes=None, peak_sign='neg',
                 unit_colors=None, figure=None, ax=None, **job_kwargs):
        BaseWidget.__init__(self, figure, ax)

        self.we = waveform_extractor
        if amplitudes is not None:
            # amplitudes must be a list of dict
            assert isinstance(amplitudes, list)
            assert all(isinstance(e, dict) for e in amplitudes)
            self.amplitudes = amplitudes
        else:
            self.amplitudes = get_spike_amplitudes(self.we, peak_sign='neg', outputs='by_unit', **job_kwargs)

        if unit_ids is None:
            unit_ids = waveform_extractor.sorting.unit_ids
        self.unit_ids = unit_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(self.we.sorting)
        self.unit_colors = unit_colors

    def plot(self):
        self._do_plot()


class AmplitudeTimeseriesWidget(AmplitudeBaseWidget):
    """
    Plots waveform amplitudes distribution.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    
    amplitudes: None or pre computed amplitudes
        If None then apmlitudes are recomputed
    
    peak_sign: 'neg', 'pos', 'both'
        In case of recomputing amplitudes.

    Returns
    -------
    W: AmplitudeDistributionWidget
        The output widget
    """

    def _do_plot(self):
        sorting = self.we.sorting
        # ~ unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()
        fs = sorting.get_sampling_frequency()

        # TODO handle segment
        ax = self.ax
        for i, unit_id in enumerate(self.unit_ids):
            for segment_index in range(num_seg):
                times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
                times = times / fs
                amps = self.amplitudes[segment_index][unit_id]
                ax.scatter(times, amps, color=self.unit_colors[unit_id], s=3, alpha=1)

                if i == 0:
                    ax.set_title(f'segment {segment_index}')
                if i == len(self.unit_ids) - 1:
                    ax.set_xlabel('Times [s]')
                if segment_index == 0:
                    ax.set_ylabel(f'{unit_id}')

        ylims = ax.get_ylim()
        if np.max(ylims) < 0:
            ax.set_ylim(min(ylims), 0)
        if np.min(ylims) > 0:
            ax.set_ylim(0, max(ylims))


class AmplitudeDistributionWidget(AmplitudeBaseWidget):
    """
    Plots waveform amplitudes distribution.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    
    amplitudes: None or pre computed amplitudes
        If None then apmlitudes are recomputed
    
    peak_sign: 'neg', 'pos', 'both'
        In case of recomputing amplitudes.

    Returns
    -------
    W: AmplitudeDistributionWidget
        The output widget
    """

    def _do_plot(self):
        sorting = self.we.sorting
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()

        ax = self.ax
        unit_amps = []
        for i, unit_id in enumerate(unit_ids):
            amps = []
            for segment_index in range(num_seg):
                amps.append(self.amplitudes[segment_index][unit_id])
            amps = np.concatenate(amps)
            unit_amps.append(amps)
        parts = ax.violinplot(unit_amps, showmeans=False, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            color = self.unit_colors[unit_ids[i]]
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.set_xticks(np.arange(len(unit_ids)) + 1)
        ax.set_xticklabels([str(unit_id) for unit_id in unit_ids])

        ylims = ax.get_ylim()
        if np.max(ylims) < 0:
            ax.set_ylim(min(ylims), 0)
        if np.min(ylims) > 0:
            ax.set_ylim(0, max(ylims))


def plot_amplitudes_timeseries(*args, **kwargs):
    W = AmplitudeTimeseriesWidget(*args, **kwargs)
    W.plot()
    return W


plot_amplitudes_timeseries.__doc__ = AmplitudeTimeseriesWidget.__doc__


def plot_amplitudes_distribution(*args, **kwargs):
    W = AmplitudeDistributionWidget(*args, **kwargs)
    W.plot()
    return W


plot_amplitudes_distribution.__doc__ = AmplitudeDistributionWidget.__doc__
