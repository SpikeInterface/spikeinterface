from .base import BaseWidget, define_widget_function_from_class
from ..core.waveform_extractor import WaveformExtractor
from ..postprocessing import compute_spike_amplitudes


class AmplitudeTimeseriesWidget(BaseWidget):
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
    compute_kwargs: dict
        Keyword arguments for computing amplitude (if not computed yet)
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None,
                 segment_index=None, compute_kwargs=None, hide_unit_selector=False,
                 backend=None, **backend_kwargs):
        sorting = waveform_extractor.sorting
        if waveform_extractor.is_extension('spike_amplitudes'):
            sac = waveform_extractor.load_extension('spike_amplitudes')
            amplitudes = sac.get_data(outputs='by_unit')
        else:
            if compute_kwargs is None:
                compute_kwargs = {}
            amplitudes = compute_spike_amplitudes(
                waveform_extractor, outputs='by_unit', **compute_kwargs)

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        if sorting.get_num_segments() > 1:
            assert segment_index is not None, "Specify segment index for multi-segment object"
        else:
            segment_index = 0
        amplitudes_segment = amplitudes[segment_index]
        total_duration = waveform_extractor.recording.get_num_samples(segment_index) / \
            waveform_extractor.recording.get_sampling_frequency()

        spiketrains_segment = {}
        for i, unit_id in enumerate(unit_ids):
            times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            times = times / sorting.get_sampling_frequency()
            spiketrains_segment[unit_id] = times

        plot_data = dict(
            amplitudes=amplitudes_segment,
            unit_ids=unit_ids,
            spiketrains=spiketrains_segment,
            total_duration=total_duration,
            hide_unit_selector=hide_unit_selector
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


plot_amplitudes_timeseries = define_widget_function_from_class(AmplitudeTimeseriesWidget, 'plot_amplitudes_timeseries')
