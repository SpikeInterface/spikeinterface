from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, NpzSortingExtractor
from spikeinterface.extractors import synthesize_random_firings


class HybridUnitsRecording(BaseRecording):

    def __init__(self, target_recording: BaseRecording, templates: np.ndarray, n_before: Union[List[int], None] = None,
                 sorting: Union[BaseSorting, None] = None, frequency: float = 10, amplitude_std: float = 0.0,
                 refrac_period: float = 2.0):
        """
        TODO
        """

        # Propagate information from the target recording.
        BaseRecording.__init__(self, target_recording.sampling_frequency, target_recording.channel_ids, target_recording.dtype)
        target_recording.copy_metadata(self)

        assert target_recording.get_num_segments() == 1 # For now, make things simple with 1 segment.

        recording_segment = HybridUnitsRecordingSegment(target_recording._recording_segments[0], templates,
                                                        n_before, sorting, frequency, amplitude_std, refrac_period)
        self.add_recording_segment(recording_segment)


class HybridUnitsRecordingSegment(BaseRecordingSegment):

    def __init__(self, target_recording: BaseRecordingSegment, templates: np.ndarray, n_before: Union[List[int], None] = None,
                 sorting: Union[BaseSorting, None] = None, frequency: float = 10, amplitude_std: float = 0.0,
                 refrac_period: float = 2.0):
    """
    TODO
    """

    self.parent_recording = target_recording
    self.templates = templates
    n_units = len(templates)

    if sorting is None: # TODO: Move out of toy_example for better spike train generation.
        t_max = recording.get_num_frames()
        fs = recording.get_sampling_frequency()

        spike_times, spike_labels = synthesize_random_firings(num_units=n_units, sampling_frequency=fs, duration=t_max)
        spike_trains = {unit_id: spike_times[spike_labels == unit_id] for unit_id in range(n_units)}

        npz_file = "TODO" # np.savez(file, **spike_trains)
        sorting = NpzSortingExtractor(npz_file)
    self.sorting = sorting

    amplitudes = {unit_id: np.random.normal(loc=1.0, scale=amplitude_std, size=len(sorting.get_unit_spike_train(unit_id)))
                  for unit_id in range(n_units)}
    self.amplitude_factor = np.where(amplitudes < 0, 0, amplitudes)


    def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None) -> np.ndarray:
        traces = self.parent_recording.get_traces(start_frame, end_frame)

        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.parent_recording.get_num_frames() if end_frame is None else end_frame
        channel_indices = list(range(len(self.parent_recording.channel_ids))) if channel_indices is None else channels_indices

        spikes, labels = self.sorting.get_all_spike_trains(outputs="unit_index")[0]
        templates_t = self.templates.shape[1]
        mask = (spikes > start_frame - templates_t) & (spikes < end_frame + templates_t) # Margins to take into account spikes outside of the frame.
        spikes = spikes[mask]
        labels = labels[mask]

        for t, unit_idx in zip(spikes, labels):
            template = self.templates[unit_idx, :, channels_indices]
            m = self.n_before[unit_idx]

            # Add template to traces
            start_traces = t - m - start_frame
            end_traces = start_traces + templates_t
            start_template = 0
            end_teplate = 0

            if start_traces < 0:
                start_template = -start_traces
                start_traces = 0
            if end_traces > end_frame - start_frame:
                end_template = end_frame - start_frame - end_traces
                end_traces = end_frame - start_frame

            traces[start_traces : end_traces] = template[start_template : end_teplate]

        return traces
