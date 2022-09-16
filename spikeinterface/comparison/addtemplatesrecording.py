from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment


class AddTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    TODO
    """

    def __init__(self, sorting: BaseSorting, templates: np.ndarray, nbefore: Union[List[int], None] = None,
                 amplitude_factor: Union[Dict[List[float]], None] = None,
                 target_recording: Union[BaseRecording, None] = None, t_max: Union[List[int], None] = None) -> None:
        
        n_units = len(sorting.unit_ids)
        assert len(templates) == unit_ids

        if nbefore is None:
            nbefore = np.argmax(np.max(np.abs(templates), axis=2), axis=1)
        else:
            assert len(nbefore) == n_units

        if amplitude_factor is None:
            amplitude_factor = {unit_id: [1.0]*len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids}

        if target_recording is not None:
            assert target_recording.get_num_segments() == sorting.get_num_segments()
            target_recording.copy_metadata(self)
        else:
            assert t_max is not None, "AddTemplatesRecording.t_max has to be set of target_recording is None."
            target_recording = None # TODO: Make recording containing only 0.

        if t_max is None:


        self.spike_vector = sorting.to_spike_vector()

        for segment_index in range(sorting.get_num_segments()):
            spikes = spike_vector[spike_vector['segment_ind'] == segment_index]
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
    amplitude_factor: Dict[List[float]]
    target_recording: Union[BaseRecordingSegment, None] = None
    t_max: int

    # def __init__(self, spike_vector: np.ndarray, templates: np.ndarray, nbefore: List[int],
    #              amplitude_factor: Dict[List[float]], target_recording: Union[BaseRecordingSegment, None] = None) -> None:

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
            traces = np.zeros([t_max, n_templates], dtype=np.int16)
        
        channel_indices = list(range(n_templates)) if channel_indices is None else channels_indices

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

            traces[start_traces : end_traces] = template[start_template : end_teplate] # multiplicator

        return traces