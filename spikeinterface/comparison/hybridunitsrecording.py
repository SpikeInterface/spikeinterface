import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, NpzSortingExtractor
from spikeinterface.extractors import synthesize_random_firings


class HybridUnitsRecording(BaseRecording):

    def __init__(self, target_recording: BaseRecording, templates: np.ndarray, n_before=None, sorting=None,
                 frequency: float = 10, amplitude_std: float = 0.0, refrac_period: float = 2.0):
        """
        TODO
        """

        assert target_recording.get_num_segments() == 1 # For now, make things simple with 1 segment.

        recording_segment = HybridUnitsRecordingSegment(target_recording._recording_segments[0], templates,
                                                        n_before, sorting, frequency, amplitude_std, refrac_period)
        self.add_recording_segment(recording_segment)


class HybridUnitsRecordingSegment(BaseRecordingSegment):

    def __init__(self, target_recording: BaseRecordingSegment, templates: np.ndarray, n_before=None, sorting=None,
                 frequency: float = 10, amplitude_std: float = 0.0, refrac_period: float = 2.0):
    """
    TODO
    """

    n_units = len(templates)

    if sorting is None: # TODO: Move out of toy_example for better spike train generation.
        t_max = recording.get_num_frames()
        fs = recording.get_sampling_frequency()

        spike_times, spike_labels = synthesize_random_firings(num_units=n_units, sampling_frequency=fs, duration=t_max)
        spike_trains = {unit_id: spike_times[spike_labels == unit_id] for unit_id in range(n_units)}

        npz_file = "TODO" # np.savez(file, **spike_trains)
        sorting = NpzSortingExtractor(npz_file)

    self.amplitude_factor = {unit_id: np.random.normal(loc=0.0, scale=amplitude_std, size=len(sorting.get_unit_spike_train(unit_id)))
                             for unit_id in range(n_units)}


    def get_traces():
        pass
