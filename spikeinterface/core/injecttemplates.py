import math
from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class, check_json


class InjectTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting: BaseSorting
        Sorting object containing all the units and their spike train.
    templates: np.ndarray[n_units, n_samples, n_channels]
        Array containing the templates to inject for all the units.
    nbefore: list[int] | int | None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    amplitude_factor: list[list[float]] | list[float] | float
        The amplitude of each spike for each unit (1.0=default).
        Can be sent as a list[float] the same size as the spike vector.
        Will default to 1.0 everywhere.
    parent_recording: BaseRecording | None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples: list[int] | int | None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.

    Returns
    -------
    injected_recording: InjectTemplatesRecording
        The recording with the templates injected.
    """

    def __init__(self, sorting: BaseSorting, templates: np.ndarray, nbefore: Union[List[int], int, None] = None,
                 amplitude_factor: Union[List[List[float]], List[float], float] = 1.0,
                 parent_recording: Union[BaseRecording, None] = None, num_samples: Union[List[int], None] = None) -> None:
        templates = np.array(templates)
        self._check_templates(templates)

        channel_ids = parent_recording.channel_ids if parent_recording is not None else list(range(templates.shape[2]))
        dtype = parent_recording.dtype if parent_recording is not None else templates.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)
        
        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            nbefore = np.argmax(np.max(np.abs(templates), axis=2), axis=1)
        elif isinstance(nbefore, (int, np.integer)):
            nbefore = [nbefore]*n_units 
        else:
            assert len(nbefore) == n_units

        if isinstance(amplitude_factor, float):
            amplitude_factor = np.array([1.0]*len(self.spike_vector), dtype=np.float32)
        elif len(amplitude_factor) != len(self.spike_vector):  # In this case, it's a list of list for amplitude by unit by spike.
            tmp = np.array([], dtype=np.float32)

            for segment_index in range(sorting.get_num_segments()):
                spike_times = [sorting.get_unit_spike_train(unit_id, segment_index=segment_index) for unit_id in sorting.unit_ids]
                spike_times = np.concatenate(spike_times)
                spike_amplitudes = np.concatenate(amplitude_factor[segment_index])

                order = np.argsort(spike_times)
                tmp = np.append(tmp, spike_amplitudes[order])

            amplitude_factor = tmp

        if parent_recording is not None:
            assert parent_recording.get_num_segments() == sorting.get_num_segments()
            assert parent_recording.get_sampling_frequency() == sorting.get_sampling_frequency()
            assert parent_recording.get_num_channels() == templates.shape[2]
            parent_recording.copy_metadata(self)

        if num_samples is None:
            if parent_recording is None:
                num_samples = [self.spike_vector['sample_ind'][-1] + templates.shape[1]]
            else:
                num_samples = [parent_recording.get_num_frames(segment_index) for segment_index in range(sorting.get_num_segments())]
        if isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]


        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector['segment_ind'], segment_index, side="left")
            end = np.searchsorted(self.spike_vector['segment_ind'], segment_index, side="right")
            spikes = self.spike_vector[start : end]

            parent_recording_segment = None if parent_recording is None else parent_recording._recording_segments[segment_index]
            recording_segment = InjectTemplatesRecordingSegment(self.sampling_frequency, self.dtype, spikes, templates, nbefore,
                                                             amplitude_factor[start:end], parent_recording_segment, num_samples[segment_index])
            self.add_recording_segment(recording_segment)

        self._kwargs = {
            "sorting": sorting,
            "templates": templates.tolist(),
            "nbefore": nbefore,
            "amplitude_factor": amplitude_factor
        }
        if parent_recording is None:
            self._kwargs['num_samples'] = num_samples
        else:
            self._kwargs['parent_recording'] = parent_recording
        self._kwargs = check_json(self._kwargs)


    @staticmethod
    def _check_templates(templates: np.ndarray):
        max_value = np.max(np.abs(templates))
        threshold = 0.01 * max_value

        if max(np.max(np.abs(templates[:, 0])), np.max(np.abs(templates[:, -1]))) > threshold:
            raise Exception("Warning!\nYour templates do not go to 0 on the edges in InjectTemplatesRecording.__init__\nPlease make your window bigger.")



class InjectTemplatesRecordingSegment(BaseRecordingSegment):

    def __init__(self, sampling_frequency: float, dtype, spike_vector: np.ndarray, templates: np.ndarray, nbefore: List[int],
                 amplitude_factor: List[List[float]], parent_recording_segment: Union[BaseRecordingSegment, None] = None, num_samples: Union[int, None] = None) -> None:

        BaseRecordingSegment.__init__(self, sampling_frequency, t_start=0 if parent_recording_segment is None else parent_recording_segment.t_start)
        assert not (parent_recording_segment is None and num_samples is None)

        self.dtype = dtype
        self.spike_vector = spike_vector
        self.templates = templates
        self.nbefore = nbefore
        self.amplitude_factor = amplitude_factor
        self.parent_recording = parent_recording_segment
        self.num_samples = parent_recording_segment.get_num_frames() if num_samples is None else num_samples

    def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None) -> np.ndarray:
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.num_samples if end_frame is None else end_frame
        channel_indices = list(range(self.templates.shape[2])) if channel_indices is None else channel_indices
        if isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.templates.shape[2]
            start = channel_indices.start if channel_indices.start is not None else 0
            step = channel_indices.step if channel_indices.step is not None else 1
            n_channels = math.ceil((stop-start) / step)
        else:
            n_channels = len(channel_indices)

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame, channel_indices).copy()
        else:
            traces = np.zeros([end_frame - start_frame, n_channels], dtype=self.dtype)

        start = np.searchsorted(self.spike_vector['sample_ind'], start_frame - self.templates.shape[1], side="left")
        end   = np.searchsorted(self.spike_vector['sample_ind'], end_frame   + self.templates.shape[1], side="right")

        for i in range(start, end):
            spike = self.spike_vector[i]
            t = spike['sample_ind']
            unit_ind = spike['unit_ind']
            template = self.templates[unit_ind][:, channel_indices]

            start_traces = t - self.nbefore[unit_ind] - start_frame
            end_traces = start_traces + template.shape[0]
            if start_traces >= end_frame-start_frame or end_traces <= 0:
                continue

            start_template = 0
            end_template = template.shape[0]

            if start_traces < 0:
                start_template = -start_traces
                start_traces = 0
            if end_traces > end_frame - start_frame:
                end_template = template.shape[0] + end_frame - start_frame - end_traces
                end_traces = end_frame - start_frame

            traces[start_traces : end_traces] += (template[start_template : end_template].astype(np.float64) * self.amplitude_factor[i]).astype(traces.dtype)

        return traces.astype(self.dtype)

    def get_num_samples(self) -> int:
        return self.num_samples


inject_templates = define_function_from_class(source_class=InjectTemplatesRecording, name="inject_templates")
