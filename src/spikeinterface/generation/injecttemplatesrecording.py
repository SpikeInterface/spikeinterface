from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Union, List, Optional

from ..core import BaseRecording, BaseRecordingSegment, BaseSorting
from ..core.core_tools import define_function_from_class


class InjectTemplatesRecording(BaseRecording):
    """
    Class for creating a recording based on spike timings and templates.
    Can be just the templates or can add to an already existing recording.

    Parameters
    ----------
    sorting: BaseSorting
        Sorting object containing all the units and their spike train.
    templates: np.ndarray[n_units, n_samples, n_channels] or np.ndarray[n_units, n_samples, n_oversampling]
        Array containing the templates to inject for all the units.
        Shape can be:
            * (num_units, num_samples, num_channels): standard case
            * (num_units, num_samples, num_channels, upsample_factor): case with oversample template to introduce sampling jitter.
    nbefore: list[int] | int | None, default: None
        Where is the center of the template for each unit?
        If None, will default to the highest peak.
    amplitude_factor: list[float] | float | None, default: None
        The amplitude of each spike for each unit.
        Can be None (no scaling).
        Can be scalar all spikes have the same factor (certainly useless).
        Can be a vector with same shape of spike_vector of the sorting.
    parent_recording: BaseRecording | None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples: list[int] | int | None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.
    upsample_vector: np.array or None, default: None.
        When templates is 4d we can simulate a jitter.
        Optional the upsample_vector is the jitter index with a number per spike in range 0-templates.sahpe[3]

    Returns
    -------
    injected_recording: InjectTemplatesRecording
        The recording with the templates injected.
    """

    def __init__(
        self,
        sorting: BaseSorting,
        templates: np.ndarray,
        nbefore: Union[List[int], int, None] = None,
        amplitude_factor: Union[List[List[float]], List[float], float, None] = None,
        parent_recording: Union[BaseRecording, None] = None,
        num_samples: Optional[List[int]] = None,
        upsample_vector: Union[List[int], None] = None,
        check_borders: bool = False,
    ) -> None:
        templates = np.asarray(templates)
        # TODO: this should be external to this class. It is not the responsability of this class to check the templates
        if check_borders:
            self._check_templates(templates)
            # lets test this only once so force check_borders=False for kwargs
            check_borders = False
        self.templates = templates

        channel_ids = parent_recording.channel_ids if parent_recording is not None else list(range(templates.shape[2]))
        dtype = parent_recording.dtype if parent_recording is not None else templates.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)

        # Important : self._serializability is not change here because it will depend on the sorting parents itself.

        n_units = len(sorting.unit_ids)
        assert len(templates) == n_units
        self.spike_vector = sorting.to_spike_vector()

        if nbefore is None:
            # take the best peak of all template
            nbefore = np.argmax(np.max(np.abs(templates), axis=(0, 2)), axis=0)

        if templates.ndim == 3:
            # standard case
            upsample_factor = None
        elif templates.ndim == 4:
            # handle also upsampling and jitter
            upsample_factor = templates.shape[3]
        elif templates.ndim == 5:
            # handle also drift
            raise NotImplementedError("Drift will be implented soon...")
            # upsample_factor = templates.shape[3]
        else:
            raise ValueError("templates have wrong dim should 3 or 4")

        if upsample_factor is not None:
            assert upsample_vector is not None
            assert upsample_vector.shape == self.spike_vector.shape

        if amplitude_factor is None:
            amplitude_vector = None
        elif np.isscalar(amplitude_factor):
            amplitude_vector = np.full(self.spike_vector.size, amplitude_factor, dtype="float32")
        else:
            amplitude_factor = np.asarray(amplitude_factor)
            assert amplitude_factor.shape == self.spike_vector.shape
            amplitude_vector = amplitude_factor

        if parent_recording is not None:
            assert parent_recording.get_num_segments() == sorting.get_num_segments()
            assert parent_recording.get_sampling_frequency() == sorting.get_sampling_frequency()
            assert parent_recording.get_num_channels() == templates.shape[2]
            parent_recording.copy_metadata(self)

        if num_samples is None:
            if parent_recording is None:
                num_samples = [self.spike_vector["sample_index"][-1] + templates.shape[1]]
            else:
                num_samples = [
                    parent_recording.get_num_frames(segment_index)
                    for segment_index in range(sorting.get_num_segments())
                ]
        elif isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]

        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            spikes = self.spike_vector[start:end]
            amplitude_vec = amplitude_vector[start:end] if amplitude_vector is not None else None
            upsample_vec = upsample_vector[start:end] if upsample_vector is not None else None

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectTemplatesRecordingSegment(
                self.sampling_frequency,
                self.dtype,
                spikes,
                templates,
                nbefore,
                amplitude_vec,
                upsample_vec,
                parent_recording_segment,
                num_samples[segment_index],
            )
            self.add_recording_segment(recording_segment)

        if not sorting.check_serializability("json"):
            self._serializability["json"] = False
        if parent_recording is not None:
            if not parent_recording.check_serializability("json"):
                self._serializability["json"] = False

        self._kwargs = {
            "sorting": sorting,
            "templates": templates.tolist(),
            "nbefore": nbefore,
            "amplitude_factor": amplitude_factor,
            "upsample_vector": upsample_vector,
            "check_borders": check_borders,
        }
        if parent_recording is None:
            self._kwargs["num_samples"] = num_samples
        else:
            self._kwargs["parent_recording"] = parent_recording

    @staticmethod
    def _check_templates(templates: np.ndarray):
        max_value = np.max(np.abs(templates))
        threshold = 0.01 * max_value

        if max(np.max(np.abs(templates[:, 0])), np.max(np.abs(templates[:, -1]))) > threshold:
            warnings.warn(
                "Warning! Your templates do not go to 0 on the edges in InjectTemplatesRecording. Please make your window bigger."
            )


class InjectTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        sampling_frequency: float,
        dtype,
        spike_vector: np.ndarray,
        templates: np.ndarray,
        nbefore: int,
        amplitude_vector: Union[List[float], None],
        upsample_vector: Union[List[float], None],
        parent_recording_segment: Union[BaseRecordingSegment, None] = None,
        num_samples: Union[int, None] = None,
    ) -> None:
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency,
            t_start=0 if parent_recording_segment is None else parent_recording_segment.t_start,
        )
        assert not (parent_recording_segment is None and num_samples is None)

        self.dtype = dtype
        self.spike_vector = spike_vector
        self.templates = templates
        self.nbefore = nbefore
        self.amplitude_vector = amplitude_vector
        self.upsample_vector = upsample_vector
        self.parent_recording = parent_recording_segment
        self.num_samples = parent_recording_segment.get_num_frames() if num_samples is None else num_samples

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.num_samples if end_frame is None else end_frame

        if channel_indices is None:
            n_channels = self.templates.shape[2]
        elif isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.templates.shape[2]
            start = channel_indices.start if channel_indices.start is not None else 0
            step = channel_indices.step if channel_indices.step is not None else 1
            n_channels = math.ceil((stop - start) / step)
        else:
            n_channels = len(channel_indices)

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame, channel_indices).copy()
        else:
            traces = np.zeros([end_frame - start_frame, n_channels], dtype=self.dtype)

        start = np.searchsorted(self.spike_vector["sample_index"], start_frame - self.templates.shape[1], side="left")
        end = np.searchsorted(self.spike_vector["sample_index"], end_frame + self.templates.shape[1], side="right")

        for i in range(start, end):
            spike = self.spike_vector[i]
            t = spike["sample_index"]
            unit_ind = spike["unit_index"]
            if self.upsample_vector is None:
                template = self.templates[unit_ind]
            else:
                upsample_ind = self.upsample_vector[i]
                template = self.templates[unit_ind, :, :, upsample_ind]

            if channel_indices is not None:
                template = template[:, channel_indices]

            start_traces = t - self.nbefore - start_frame
            end_traces = start_traces + template.shape[0]
            if start_traces >= end_frame - start_frame or end_traces <= 0:
                continue

            start_template = 0
            end_template = template.shape[0]

            if start_traces < 0:
                start_template = -start_traces
                start_traces = 0
            if end_traces > end_frame - start_frame:
                end_template = template.shape[0] + end_frame - start_frame - end_traces
                end_traces = end_frame - start_frame

            wf = template[start_template:end_template]
            if self.amplitude_vector is not None:
                wf = wf * self.amplitude_vector[i]
            traces[start_traces:end_traces] += wf

        return traces.astype(self.dtype, copy=False)

    def get_num_samples(self) -> int:
        return self.num_samples


inject_templates = define_function_from_class(source_class=InjectTemplatesRecording, name="inject_templates")
