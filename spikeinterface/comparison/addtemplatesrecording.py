from dataclasses import dataclass
from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment


class AddTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting: BaseSorting
        Sorting object containing all the units and their spike train.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units.
    nbefore: list[int] | None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    amplitude_factor: list[list[float]] | None
        The amplitude of each spike for each unit (1.0=default).
        If None, will default to 1.0 everywhere.
    target_recording: BaseRecording | None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    t_max: list[int] | None
        The number of frames in the recording.
    """

    def __init__(self, sorting: BaseSorting, templates: np.ndarray, nbefore: Union[List[int], None] = None,
                 amplitude_factor: Union[List[List[float]], None] = None,
                 target_recording: Union[BaseRecording, None] = None, t_max: Union[List[int], None] = None) -> None:
        channel_ids = target_recording.channel_ids if target_recording is not None else list(range(templates.shape[2]))
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, templates.dtype)
        
        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            nbefore = np.argmax(np.max(np.abs(templates), axis=2), axis=1)
        else:
            assert len(nbefore) == n_units

        if amplitude_factor is None:
            amplitude_factor = [[1.0]*len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids]

        if target_recording is not None:
            assert target_recording.get_num_segments() == sorting.get_num_segments()
            target_recording.copy_metadata(self)

        if t_max is None:
            if target_recording is None:
                t_max = [self.spike_vector['sample_ind'][-1] + templates.shape[1]]
            else:
                t_max = [target_recording.get_num_frames(segment_index) for segment_index in range(sorting.get_num_segments())]


        for segment_index in range(sorting.get_num_segments()):
            spikes = self.spike_vector[self.spike_vector['segment_ind'] == segment_index]
            target_recording_segment = None if target_recording is None else target_recording._recording_segments[segment_index]
            recording_segment = AddTemplatesRecordingSegment(spikes, templates, nbefore, amplitude_factor,
                                                             target_recording_segment, t_max[segment_index])
            self.add_recording_segment(recording_segment)



@dataclass
class AddTemplatesRecordingSegment(BaseRecordingSegment):
    """
    TODO
    """

    spike_vector: np.ndarray
    templates: np.ndarray
    nbefore: List[int]
    amplitude_factor: List[List[float]]
    t_max: int
    target_recording: Union[BaseRecordingSegment, None] = None

    # def __init__(self, spike_vector: np.ndarray, templates: np.ndarray, nbefore: List[int],
    #              amplitude_factor: List[List[float]], target_recording: Union[BaseRecordingSegment, None] = None) -> None:

    #     self.spike_vector = spike_vector
    #     self.templates = templates
    #     self.nbefore = nbefore
    #     self.amplitude_factor = amplitude_factor
    #     self.target_recording = target_recording

    def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None) -> np.ndarray:
        n_channels = self.templates.shape[1]
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.t_max if end_frame is None else end_frame

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame)
        else:
            traces = np.zeros([t_max, n_channels], dtype=np.int16) # TODO: dtype
        
        channel_indices = list(range(n_channels)) if channel_indices is None else channels_indices

        for spike in self.spike_vector:
            t = spike['sample_ind']
            unit_ind = spike['unit_ind']

            template = self.templates[unit_ind, :, channels_indices]
            m = self.nbefore[unit_ind]

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

            traces[start_traces : end_traces] = template[start_template : end_teplate] * self.amplitude_factor[unit_ind]

        return traces