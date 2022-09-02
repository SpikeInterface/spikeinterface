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
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None, unit_colors=None,
                 segment_index=None, hide_unit_selector=False, plot_histograms=False,
                 bins=None, backend=None, **backend_kwargs):
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
        
        if plot_histograms and bins is None:
            bins = 100

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            amplitudes=amplitudes_segment,
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            spiketrains=spiketrains_segment,
            total_duration=total_duration,
            plot_histograms=plot_histograms,
            bins=bins,
            hide_unit_selector=hide_unit_selector
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)



