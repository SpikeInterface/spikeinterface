import numpy as np

from .base import BaseWidget
from .utils import get_some_colors

from ..core.waveform_extractor import WaveformExtractor
from ..postprocessing import compute_spike_amplitudes


class AmplitudesWidget(BaseWidget):
    """
    Plots spike amplitudes

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The input waveform extractor
    unit_ids: list
        List of unit ids.
    segment_index: int
        The segment index (or None if mono-segment)
    max_spikes_per_unit: int
        Number of max spikes per unit to display. Use None for all spikes.
        Default 500.
    hide_unit_selector : bool
        If True the unit selector is not displayed
        (sortingview backend)
    plot_histogram : bool
        If True, an histogram of the amplitudes is plotted on the right axis 
        (matplotlib backend)
    bins : int
        If plot_histogram is True, the number of bins for the amplitude histogram.
        If None (default), this is automatically adjusted.
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None, unit_colors=None,
                 segment_index=None, max_spikes_per_unit=500, hide_unit_selector=False, 
                 plot_histograms=False, bins=None, backend=None, **backend_kwargs):
        sorting = waveform_extractor.sorting
        self.check_extensions(waveform_extractor, "spike_amplitudes")
        sac = waveform_extractor.load_extension('spike_amplitudes')
        amplitudes = sac.get_data(outputs='by_unit')


        if unit_ids is None:
            unit_ids = sorting.unit_ids
    
        if unit_colors is None:
            unit_colors = get_some_colors(sorting.unit_ids)

        if sorting.get_num_segments() > 1:
            assert segment_index is not None, "Specify segment index for multi-segment object"
        else:
            segment_index = 0
        amplitudes_segment = amplitudes[segment_index]
        total_duration = waveform_extractor.recording.get_num_samples(segment_index) / \
            waveform_extractor.recording.get_sampling_frequency()
        
        spiketrains_segment = {}
        for i, unit_id in enumerate(sorting.unit_ids):
            times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            times = times / sorting.get_sampling_frequency()
            spiketrains_segment[unit_id] = times

        all_spiketrains = spiketrains_segment
        all_amplitudes = amplitudes_segment
        if max_spikes_per_unit is not None:
            spiketrains_to_plot = dict()
            amplitudes_to_plot = dict()
            for unit, st in all_spiketrains.items():
                amps = all_amplitudes[unit]
                if len(st) > max_spikes_per_unit:
                    random_idxs = np.random.permutation(len(st))[:max_spikes_per_unit]
                    spiketrains_to_plot[unit] = st[random_idxs]
                    amplitudes_to_plot[unit] = amps[random_idxs]
                else:
                    spiketrains_to_plot[unit] = st
                    amplitudes_to_plot[unit] = amps
        else:
            spiketrains_to_plot = all_spiketrains
            amplitudes_to_plot = all_amplitudes

        if plot_histograms and bins is None:
            bins = 100

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            amplitudes=amplitudes_to_plot,
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            spiketrains=spiketrains_to_plot,
            total_duration=total_duration,
            plot_histograms=plot_histograms,
            bins=bins,
            hide_unit_selector=hide_unit_selector
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)



