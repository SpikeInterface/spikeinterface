from typing import Union, Optional, List, Literal

import numpy as np
from spikeinterface.core import Templates, BaseRecording, BaseSorting, BaseRecordingSegment
import math


def interpolate_templates(templates_array, source_locations, dest_locations, interpolation_method="cubic"):
    """
    Interpolate templates_array to new positions.
    Useful to create motion or to remap templates_array form probeA to probeB.

    Note that several moves can be done by broadcasting when dest_locations have more than one dimension.
    
    Parameters
    ----------
    templates_array: np.array
        A numpy array with dense templates_array.
        shape = (num_template, num_sample, num_channel)
    source_locations: np.array 
        The channel source location corresponding to templates_array.
        shape = (num_channel, 2)
    dest_locations: np.array
        The new channel position, if ndim == 3, then the interpolation is broadcated with last dim.
        shape = (num_channel, 2) or (num_motion, num_channel, 2)
    interpolation_method: str, default "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates_array: np.array
        shape = (num_template, num_sample, num_channel) or = (num_motion, num_template, num_sample, num_channel, )
    """
    import scipy.interpolate

    source_locations = np.asarray(source_locations)
    dest_locations = np.asarray(dest_locations)
    if dest_locations.ndim == 2:
        new_shape = templates_array.shape
    elif dest_locations.ndim == 3:
        new_shape = (dest_locations.shape[0], ) + templates_array.shape
    else:
        raise ValueError(f"Incorrect dimensions for dest_locations: {dest_locations.ndim}. Dimension can be 2 or 3. ")

    new_templates_array = np.zeros(new_shape, dtype=templates_array.dtype)
    
    for template_ind in range(templates_array.shape[0]):
        for time_ind in range(templates_array.shape[1]):
            template = templates_array[template_ind, time_ind, :]
            interp_template = scipy.interpolate.griddata(source_locations, template, dest_locations,
                                                         method=interpolation_method,
                                                         fill_value=0)
            if dest_locations.ndim == 2:
                new_templates_array[template_ind, time_ind, :] = interp_template
            elif dest_locations.ndim == 3: 
                new_templates_array[:, template_ind, time_ind, :] = interp_template

    return new_templates_array




def move_dense_templates(templates_array, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates_array given some displacements using spatial interpolation (cubic or linear).
    Optionally can be remapped to another probe with a different geometry.
    
    This function operates on dense template.
    
    Note : in this function no checks are done to see if templates_array can be interpolatable after displacements.
    To check if the given displacements are interpolatable use the higher level function move_templates().
    
    Parameters
    ----------
    templates_array: np.array
        A numpy array with dense templates_array.
        shape = (num_template, num_sample, num_channel)
    displacements: np.array
        Displacement vector
        shape: (num_displacement, 2)
    source_probe: Probe
        The Probe object on which templates_array are defined
    dest_probe: Probe | None, default: None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method: "cubic" | "linear", default: "cubic"
        The interpolation method.        
    
    Returns
    -------
    new_templates_array: np.array
        shape = (num_displacement, num_template, num_sample, num_channel, )
    """
    assert displacements.ndim == 2
    assert displacements.shape[1] == 2

    if dest_probe is None:
        dest_probe = source_probe
    src_channel_locations = source_probe.contact_positions
    dest_channel_locations = dest_probe.contact_positions
    moved_locations = dest_channel_locations[np.newaxis, :, :] - displacements.reshape(-1, 1, 2)
    templates_array_moved = interpolate_templates(templates_array, src_channel_locations, moved_locations, interpolation_method=interpolation_method)
    return templates_array_moved




class DriftingTemplates(Templates):
    """
    Templates with drift.
    This is usefull to generate drifting recording.

    This class support 2 differents strategies:
      * move every templates on-the-fly, this lead toone interpolation per spike
      * precompute some displacements for all templates and use a discreate interpolation, for instance by step of 1um
        This is the same strategy used by MEArec

    
    """
    def __init__(self, **kwargs):
        Templates.__init__(self, **kwargs)
        assert self.probe is not None, "DriftingTemplates need a Probe in the init"

        self.all_templates_moved = None
        self.displacements = None
    
    @classmethod
    def from_static(cls, templates):
        drifting_teplates = cls(
            templates_array=templates.templates_array,
            sampling_frequency=templates.sampling_frequency,
            nbefore=templates.nbefore,
            probe=templates.probe,
        )
        return drifting_teplates

    def move_one_template(self, unit_index, displacement, **interpolation_kwargs):
        """
        Move on template.
        """
        assert displacement.shape == (1, 2)

        one_template_array = self.get_one_template_dense(unit_index)
        one_template_array = one_template_array[np.newaxis, :, :]

        template_array_moved = move_dense_templates(one_template_array, displacement, self.probe, **interpolation_kwargs)
        # one motion one template keep only (num_samples, num_channels)
        template_array_moved = template_array_moved[0, 0, :, :]

        return template_array_moved

    def precompute_displacements(self, displacements, **interpolation_kwargs):
        """
        Precompute sevral displacements for all template.
        """

        dense_static_templates = self.get_dense_templates()

        self.template_array_moved = move_dense_templates(dense_static_templates, displacements, self.probe, **interpolation_kwargs)
        self.displacements = displacements





class InjectDriftingTemplatesRecording(BaseRecording):
    """
    Class for injecting drifting templates into a recording.
    This is similar to :py:class:`InjectTemplatesRecording` but with drifts.

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
        parent_recording: Union[BaseRecording, None] = None,
        drifting_templates: DriftingTemplates=None,
        num_samples: Optional[List[int]] = None,
        amplitude_factor: Union[List[List[float]], List[float], float, None] = None,
        # TODO handle upsample vector
        # upsample_vector: Union[List[int], None] = None,
    ):
        assert isinstance(drifting_templates, DriftingTemplates)
        self.drifting_templates = drifting_templates

        if parent_recording is None:
            assert drifting_templates.channel_ids is not None
            channel_ids = drifting_templates.channel_ids
        else:
            assert drifting_templates.sampling_frequency == parent_recording.sampling_frequency
            assert np.array_equal(drifting_templates.probe.contact_positions, parent_recording.get_channel_locations)
            channel_ids = parent_recording.channel_ids
            

        dtype = parent_recording.dtype if parent_recording is not None else self.drifting_templates_array.dtype
        BaseRecording.__init__(self, sorting.get_sampling_frequency(), channel_ids, dtype)

        assert drifting_templates.num_units == sorting.unit_ids.size
        self.spike_vector = sorting.to_spike_vector()


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
            assert parent_recording.get_num_channels() == drifting_templates.num_channels
            parent_recording.copy_metadata(self)

        if num_samples is None:
            assert parent_recording is not None
            num_samples = [
                parent_recording.get_num_frames(segment_index)
                for segment_index in range(parent_recording.get_num_segments())
            ]
        elif isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1
            num_samples = [num_samples]

        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            spikes = self.spike_vector[start:end]
            amplitude_vec = amplitude_vector[start:end] if amplitude_vector is not None else None
            # upsample_vec = upsample_vector[start:end] if upsample_vector is not None else None

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectDriftingTemplatesRecordingSegment(
                self.dtype,
                spikes,
                drifting_templates,
                amplitude_vec,
                # upsample_vec,
                parent_recording_segment,
                num_samples[segment_index],
            )
            self.add_recording_segment(recording_segment)

        # self._kwargs = {
        #     "sorting": sorting,
        #     "templates": templates.tolist(),
        #     "nbefore": nbefore,
        #     "amplitude_factor": amplitude_factor,
        #     "upsample_vector": upsample_vector,
        #     "check_borders": check_borders,
        # }
        # if parent_recording is None:
        #     self._kwargs["num_samples"] = num_samples
        # else:
        #     self._kwargs["parent_recording"] = parent_recording



class InjectDriftingTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        dtype,
        spike_vector: np.ndarray,
        drifting_templates: np.ndarray,
        amplitude_vector: Union[List[float], None],
        # upsample_vector: Union[List[float], None],
        parent_recording_segment: Union[BaseRecordingSegment, None] = None,
        num_samples: Union[int, None] = None,
    ) -> None:
        BaseRecordingSegment.__init__(
            self,
            drifting_templates.sampling_frequency,
            t_start=0 if parent_recording_segment is None else parent_recording_segment.t_start,
        )
        assert not (parent_recording_segment is None and num_samples is None)

        self.dtype = dtype
        self.spike_vector = spike_vector
        self.drifting_templates = drifting_templates
        self.nbefore = drifting_templates.nbefore
        self.amplitude_vector = amplitude_vector
        # self.upsample_vector = upsample_vector
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
            n_channels = self.drifting_templates.shape[2]
        elif isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.drifting_templates.shape[2]
            start = channel_indices.start if channel_indices.start is not None else 0
            step = channel_indices.step if channel_indices.step is not None else 1
            n_channels = math.ceil((stop - start) / step)
        else:
            n_channels = len(channel_indices)

        if self.parent_recording is not None:
            traces = self.parent_recording.get_traces(start_frame, end_frame, channel_indices).copy()
        else:
            traces = np.zeros([end_frame - start_frame, n_channels], dtype=self.dtype)

        num_samples = self.drifting_templates.num_samples
        start = np.searchsorted(self.spike_vector["sample_index"], start_frame - num_samples, side="left")
        end = np.searchsorted(self.spike_vector["sample_index"], end_frame + num_samples, side="right")

        for i in range(start, end):
            spike = self.spike_vector[i]
            t = spike["sample_index"]
            unit_ind = spike["unit_index"]
            if self.upsample_vector is None:
                template = self.drifting_templates[unit_ind]
            else:
                upsample_ind = self.upsample_vector[i]
                template = self.drifting_templates[unit_ind, :, :, upsample_ind]

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
                wf *= self.amplitude_vector[i]
            traces[start_traces:end_traces] += wf

        return traces.astype(self.dtype)

    def get_num_samples(self) -> int:
        return self.num_samples

