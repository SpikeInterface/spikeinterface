import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget, BaseMultiWidget

from ..toolkit import get_unit_amplitudes





class AmplitudeBaseWidget(BaseMultiWidget):
    def __init__(self, waveform_extractor, amplitudes=None, peak_sign='neg',
            figure=None, ax=None, axes=None, **job_kwargs):
        BaseMultiWidget.__init__(self, figure, ax, axes)
        
        self.we = waveform_extractor
        if amplitudes is not None:
            # amplitudes must be a list of dict
            assert isinstance(amplitudes, list)
            assert all(isinstance(e, dict) for e in amplitudes)
            self.amplitudes = amplitudes
        else:
            self.amplitudes = get_unit_amplitudes(self.we,  peak_sign='neg', outputs='by_units', **job_kwargs)
    
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
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()
        fs = sorting.get_sampling_frequency()
        
        nrows, ncols = len(unit_ids), num_seg
        num_ax = 0
        for i, unit_id in enumerate(unit_ids):
            for segment_index in range(num_seg):
                ax = self.get_tiled_ax(num_ax, nrows, ncols)
                times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
                times = times / fs
                amps = self.amplitudes[segment_index][unit_id]
                ax.scatter(times, amps)
                num_ax += 1
                
                if i == 0:
                    ax.set_title(f'segment {segment_index}')
                if i == len(unit_ids) - 1:
                    ax.set_xlabel('Times [s]')
                if segment_index == 0:
                    ax.set_ylabel(f'{unit_id}')
                


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
        
        nrows, ncols = len(unit_ids), num_seg
        num_ax = 0
        for segment_index in range(num_seg):
            for i, unit_id in enumerate(unit_ids):
                ax = self.get_tiled_ax(num_ax, nrows, ncols)
                amps = self.amplitudes[segment_index][unit_id]
                ax.hist(amps, density=True, color='gray', bins=10)
                num_ax += 1
                
                if i == 0:
                    ax.set_title(f'segment {segment_index}')
                if i == len(unit_ids) - 1:
                    ax.set_xlabel('Amplitudes')
                if segment_index == 0:
                    ax.set_ylabel(f'{unit_id}')


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
