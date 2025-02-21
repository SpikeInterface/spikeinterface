from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, Templates


def interpolate_templates(templates_array, source_locations, dest_locations, interpolation_method="cubic"):
    """
    Interpolate templates_array to new positions.
    Useful to create motion or to remap templates_array form probeA to probeB.

    Note that multiple interpolations can be performed by broadcasting when dest_locations have more than one dimension.

    Parameters
    ----------
    templates_array : np.array
        A numpy array with dense templates_array.
        shape = (num_templates, num_samples, num_channels)
    source_locations : np.array
        The channel source location corresponding to templates_array.
        shape = (num_channels, 2)
    dest_locations : np.array
        The new channel position, if ndim == 3, then the interpolation is broadcated with last dim.
        shape = (num_channels, 2) or (num_motions, num_channels, 2)
    interpolation_method : str, default "cubic"
        The interpolation method.

    Returns
    -------
    new_templates_array : np.array
        shape = (num_templates, num_samples, num_channels) or = (num_motions, num_templates, num_samples, num_channel)
    """
    import scipy.interpolate

    source_locations = np.asarray(source_locations)
    dest_locations = np.asarray(dest_locations)
    if dest_locations.ndim == 2:
        new_shape = (*templates_array.shape[:2], len(dest_locations))
    elif dest_locations.ndim == 3:
        new_shape = (
            dest_locations.shape[0],
            *templates_array.shape[:2],
            dest_locations.shape[1],
        )
    else:
        raise ValueError(f"Incorrect dimensions for dest_locations: {dest_locations.ndim}. Dimensions can be 2 or 3. ")

    new_templates_array = np.zeros(new_shape, dtype=templates_array.dtype)

    for template_index in range(templates_array.shape[0]):
        for sample_index in range(templates_array.shape[1]):
            template = templates_array[template_index, sample_index, :]
            interp_template = scipy.interpolate.griddata(
                source_locations, template, dest_locations, method=interpolation_method, fill_value=0
            )
            if dest_locations.ndim == 2:
                new_templates_array[template_index, sample_index, :] = interp_template
            elif dest_locations.ndim == 3:
                new_templates_array[:, template_index, sample_index, :] = interp_template

    return new_templates_array


def move_dense_templates(templates_array, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates_array given some displacements using spatial interpolation (cubic or linear).
    Optionally, the displaced templates can be remapped to another probe with a different geometry.

    This function operates on dense templates only.

    Note: in this function no checks are done to see if templates_array can be interpolatable after displacements.
    To check if the given displacements are interpolatable use the higher-level function move_templates().

    Parameters
    ----------
    templates_array : np.array
        A numpy array with dense templates_array.
        shape = (num_templates, num_samples, num_channels)
    displacements : np.array
        Displacement vector
        shape : (num_displacement, 2)
    source_probe : Probe
        The Probe object on which templates_array are defined
    dest_probe : Probe | None, default: None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method : "cubic" | "linear", default: "cubic"
        The interpolation method.

    Returns
    -------
    new_templates_array : np.array
        shape = (num_displacement, num_templates, num_samples, num_channels)
    """
    assert displacements.ndim == 2
    assert displacements.shape[1] == 2

    if dest_probe is None:
        dest_probe = source_probe
    src_channel_locations = source_probe.contact_positions
    dest_channel_locations = dest_probe.contact_positions
    moved_locations = dest_channel_locations[np.newaxis, :, :] - displacements.reshape(-1, 1, 2)
    templates_array_moved = interpolate_templates(
        templates_array, src_channel_locations, moved_locations, interpolation_method=interpolation_method
    )
    return templates_array_moved


class DriftingTemplates(Templates):
    """
    Templates with drift.
    This is useful to generate drifting recording.

    This class supports 2 different strategies:
      * move every templates on-the-fly, this lead to one interpolation per spike
      * precompute some displacements for all templates and use a discreate interpolation, for instance by step of 1um
        This is the same strategy used by MEArec.

    Parameters
    ----------
    templates_array_moved : np.array
        Shape is (num_displacement, num_templates, num_samples, num_channels)
    displacements : np.array
        Displacement vector
        shape : (num_displacement, 2)
    **static_kwargs : dict
        Keyword arguments for `Templates`
    """

    def __init__(self, templates_array_moved=None, displacements=None, **static_kwargs):
        Templates.__init__(self, **static_kwargs)
        assert self.probe is not None, "DriftingTemplates need a Probe in the init"
        if templates_array_moved is not None:
            if displacements is None:
                raise ValueError(
                    "Please pass both template_array_moved and displacements to DriftingTemplates "
                    "if you are using precomputed displaced templates."
                )
        self.templates_array_moved = templates_array_moved
        self.displacements = displacements

    @classmethod
    def from_static_templates(cls, templates: Templates):
        """
        Construct a DriftingTemplates object given static templates.
        The drifting templates can be then computed using the `precompute_displacements` method.

        Parameters
        ----------
        templates : Templates
            The static templates.

        Returns
        -------
        drifting_templates : DriftingTemplates
            The drifting templates object.

        """
        drifting_templates = cls(
            templates_array=templates.templates_array,
            sampling_frequency=templates.sampling_frequency,
            nbefore=templates.nbefore,
            probe=templates.probe,
            sparsity_mask=templates.sparsity_mask,
            is_scaled=templates.is_scaled,
            unit_ids=templates.unit_ids,
            channel_ids=templates.channel_ids,
        )
        return drifting_templates

    @classmethod
    def from_precomputed_templates(
        cls,
        templates_array_moved: ArrayLike,
        displacements: ArrayLike,
        sampling_frequency: float,
        nbefore: int,
        probe: Probe,
    ):
        """Construct a DriftingTemplates object given precomputed drifting templates

        Parameters
        ----------
        templates_array_moved : np.array
            Shape is (num_displacement, num_templates, num_samples, num_channels)
        displacements : np.array
            Shape is (num_displacement, 2). Last axis is xy, as in make_linear_displacement below.
        sampling_frequency : float
        nbefore : int
        probe : probeinterface.Probe

        Returns
        -------
        drifting_templates : DriftingTemplates
            The drifting templates object.
        """
        # take the central templates as representatives, just to make the super()
        # constructor happy. they won't be used as drifting templates.
        templates_static = templates_array_moved[templates_array_moved.shape[0] // 2]
        return cls(
            templates_array=templates_static,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=probe,
            templates_array_moved=templates_array_moved,
            displacements=displacements,
        )

    def move_one_template(self, unit_index, displacement, **interpolation_kwargs):
        """
        Move on template given a displacement.

        Parameters
        ----------
        unit_index : int
            The unit index to move.
        displacements : np.array
            The displacement vector.
            shape = (1, 2)
        **interpolation_kwargs : keyword arguments for `move_dense_templates` function

        Returns
        -------
        template_array_moved : np.array
            The moved template.
            shape = (num_displacements, num_samples, num_channels)
        """
        assert displacement.shape == (1, 2)

        one_template_array = self.get_one_template_dense(unit_index)
        one_template_array = one_template_array[np.newaxis, :, :]

        template_array_moved = move_dense_templates(
            one_template_array, displacement, self.probe, **interpolation_kwargs
        )
        # one motion one template keep only (num_samples, num_channels)
        template_array_moved = template_array_moved[0, 0, :, :]

        return template_array_moved

    def precompute_displacements(self, displacements, **interpolation_kwargs):
        """
        Precompute several displacements for all template.

        Parameters
        ----------
        displacements : np.array
            The displacement vector.
            shape = (num_displacements, 2)
        **interpolation_kwargs : keyword arguments for `move_dense_templates` function
        """
        dense_static_templates = self.get_dense_templates()

        self.templates_array_moved = move_dense_templates(
            dense_static_templates, displacements, self.probe, **interpolation_kwargs
        )
        self.displacements = displacements


def make_linear_displacement(start, stop, num_step=10):
    """
    Generates 2D linear displacements between `start` and `stop` positions (included in returned displacements).

    Parameters
    ----------
    start : np.array of 2 elements
        The start position.
    stop : np.array of 2 elements
        The stop position.
    num_step : int, default: 10
        The number of steps between start and stop.

    Returns
    -------
    displacements : np.array
        The displacements with shape (num_step, 2)
    """
    if num_step < 1:
        raise ValueError("make_linear_displacement needs num_step > 0")
    if num_step == 1:
        displacements = ((start + stop) / 2)[np.newaxis, :]
    else:
        displacements = (stop[np.newaxis, :] - start[np.newaxis, :]) / (num_step - 1) * np.arange(num_step)[
            :, np.newaxis
        ] + start[np.newaxis, :]
    return displacements


class InjectDriftingTemplatesRecording(BaseRecording):
    """
    Class for injecting drifting templates into a recording.
    This is similar to :py:class:`InjectTemplatesRecording` but with drifts.

    Parameters
    ----------
    sorting : BaseSorting
        Sorting object containing all the units and their spike train
    drifting_templates : DriftingTemplates
        The drifting template object
    displacement_vectors : list of numpy array
        The lenght of the list is the number of segment.
        Per segment, the drift vector is a numpy array with shape (num_times, 2, num_motions)
        num_motions is generally = 1 but can be > 1 in case of combining several drift vectors
    displacement_sampling_frequency : float
        The sampling frequency of drift vector
    displacement_unit_factor : numpy array or None, default: None
        A array containing the factor per unit of the drift.
        This is used to create non rigid with a factor gradient of depending on units position.
        shape (num_units, num_motions)
        If None then all unit have the same factor (1) and the drift is rigid.
    parent_recording : BaseRecording or None, default: None
        The recording over which to add the templates.
        If None, will default to traces containing all 0.
    num_samples : list[int] or int or None, default: None
        The number of samples in the recording per segment.
        You can use int for mono-segment objects.
    amplitude_factor : list of numpy array or numpy array or float or None, default: None
        Controls the amplitude scaling for each spike for each unit.
        If None, no amplitude scaling is applied.
        If scalar all spikes have the same factor (certainly useless).
        If vector, it must have the same size as the spike vector.
    mode : str, default: "precompute"
        Mode for how to compute templates.

    Returns
    -------
    injected_recording : InjectDriftingTemplatesRecording
        The recording with the templates injected.
    """

    def __init__(
        self,
        sorting: BaseSorting,
        drifting_templates: DriftingTemplates,
        displacement_vectors: list[np.ndarray],
        displacement_sampling_frequency: float,
        displacement_unit_factor: Optional[np.ndarray] = None,
        parent_recording: Optional[BaseRecording] = None,
        num_samples: Optional[list[int]] = None,
        amplitude_factor: list[np.ndarray] | np.ndarray | float | None = None,
        mode="precompute",
        # TODO handle upsample vector
        # upsample_vector: Union[list[int], None] = None,
    ):
        import scipy.spatial

        assert isinstance(
            drifting_templates, DriftingTemplates
        ), "drifting_templates must be a DriftingTemplates object"
        self.drifting_templates = drifting_templates

        if parent_recording is None:
            assert drifting_templates.channel_ids is not None
            channel_ids = drifting_templates.channel_ids
        else:
            assert drifting_templates.sampling_frequency == parent_recording.sampling_frequency
            channel_ids = parent_recording.channel_ids

        dtype = (
            parent_recording.dtype if parent_recording is not None else self.drifting_templates.templates_array.dtype
        )
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
            assert (
                parent_recording.get_num_segments() == sorting.get_num_segments()
            ), "parent_recording must have the same number of segments as sorting"
            assert (
                parent_recording.get_sampling_frequency() == sorting.get_sampling_frequency()
            ), "parent_recording must have the same sampling frequency as sorting"
            assert (
                parent_recording.get_num_channels() == drifting_templates.num_channels
            ), "parent_recording must have the same number of channels as templates"
            parent_recording.copy_metadata(self)

        if num_samples is None:
            assert parent_recording is not None, "num_samples must be provided if parent_recording is None"
            num_samples = [
                parent_recording.get_num_frames(segment_index)
                for segment_index in range(parent_recording.get_num_segments())
            ]
        elif isinstance(num_samples, int):
            assert sorting.get_num_segments() == 1, "num_samples must be a list if sorting has several segments"
            num_samples = [num_samples]

        # check drift vector shape and length
        assert (
            len(displacement_vectors) == sorting.get_num_segments()
        ), "displacement_vectors must have the same number of segments as sorting"
        if displacement_unit_factor is None:
            displacement_unit_factor = np.ones(sorting.get_num_units(), dtype="float32")
        assert displacement_unit_factor.shape[0] == len(
            sorting.unit_ids
        ), "displacement_unit_factor must have the same number of units as sorting"
        # displacement_vectors_indices = []
        for num_segment in range(sorting.get_num_segments()):
            duration = displacement_vectors[num_segment].shape[0] / displacement_sampling_frequency
            assert (
                duration >= num_samples[num_segment] / sorting.get_sampling_frequency()
            ), "displacement_vectors duration must span the entire recording segment"
            assert (
                displacement_vectors[num_segment].shape[2] == displacement_unit_factor.shape[1]
            ), "displacement_vectors and displacement_unit_factor must have the same number of motions"

        # TODO SharedMem for templates
        if mode == "precompute":
            assert (
                drifting_templates.templates_array_moved is not None
            ), "drifting_templates must have precomputed displacements"
            displacements = drifting_templates.displacements

            # compute the displacement indicies
            segment_slices = []
            displacement_indices = np.zeros(self.spike_vector.size, dtype="int64")
            for segment_index in range(sorting.get_num_segments()):
                start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
                end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
                segment_slices.append((start, end))

                spike_vector_seg = self.spike_vector[start:end]

                displacement_vecs = displacement_vectors[segment_index]

                # bin index in the displacement_vecs.shape[0] (time)
                time_bin = (
                    spike_vector_seg["sample_index"] / sorting.sampling_frequency * displacement_sampling_frequency
                ).astype("int64")

                # each spike is the linear sum of several displacement
                # this is (num_spike, 2)
                factors = displacement_unit_factor[spike_vector_seg["unit_index"]]
                summed_displacement = np.sum(displacement_vecs[time_bin] * factors[:, np.newaxis, :], axis=2)

                # we go to indices by the nearest precomputed displacements
                # this is (num_spike, ) relate to indices
                inds = np.argmin(scipy.spatial.distance.cdist(displacements, summed_displacement), axis=0)
                # just by paranoia
                inds = np.clip(inds, 0, displacements.shape[0] - 1)
                # this also cast to int64
                displacement_indices[start:end] = inds

        # recording segment
        for segment_index in range(sorting.get_num_segments()):
            start = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="left")
            end = np.searchsorted(self.spike_vector["segment_index"], segment_index, side="right")
            start, end = segment_slices[segment_index]

            amplitude_vec = amplitude_vector[start:end] if amplitude_vector is not None else None
            # upsample_vec = upsample_vector[start:end] if upsample_vector is not None else None

            parent_recording_segment = (
                None if parent_recording is None else parent_recording._recording_segments[segment_index]
            )
            recording_segment = InjectDriftingTemplatesRecordingSegment(
                self.dtype,
                self.spike_vector[start:end],
                drifting_templates,
                amplitude_vec,
                # upsample_vec,
                parent_recording_segment,
                num_samples[segment_index],
                displacement_indices[start:end],
                drifting_templates.templates_array_moved,
            )
            self.add_recording_segment(recording_segment)

        self.set_probe(drifting_templates.probe, in_place=True)

        # templates are too large, we don't serialize them to JSON
        self._serializability["json"] = False

        self._kwargs = {
            "sorting": sorting,
            "drifting_templates": drifting_templates,
            "displacement_vectors": displacement_vectors,
            "displacement_sampling_frequency": displacement_sampling_frequency,
            "displacement_unit_factor": displacement_unit_factor,
            "amplitude_factor": amplitude_factor,
            "mode": mode,
        }
        if parent_recording is None:
            self._kwargs["num_samples"] = num_samples
        else:
            self._kwargs["parent_recording"] = parent_recording


class InjectDriftingTemplatesRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        dtype,
        spike_vector: np.ndarray,
        drifting_templates: DriftingTemplates,
        amplitude_vector: Optional[np.ndarray] = None,
        parent_recording_segment: Optional[BaseRecordingSegment] = None,
        num_samples: Optional[int] = None,
        displacement_indices: Optional[np.ndarray] = None,
        templates_array_moved: Optional[np.ndarray] = None,
        # upsample_vector: Union[list[float], None],
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
        # TODO: self.upsample_vector = upsample_vector
        self.upsample_vector = None
        self.parent_recording = parent_recording_segment
        self.num_samples = parent_recording_segment.get_num_samples() if num_samples is None else num_samples
        self.num_samples = int(num_samples)

        self.displacement_indices = displacement_indices
        self.templates_array_moved = templates_array_moved

    def get_traces(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        channel_indices: Optional[list] = None,
    ) -> np.ndarray:
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.num_samples if end_frame is None else end_frame

        if channel_indices is None:
            n_channels = self.drifting_templates.num_channels
        elif isinstance(channel_indices, slice):
            stop = channel_indices.stop if channel_indices.stop is not None else self.drifting_templates.num_channels
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
            unit_index = spike["unit_index"]
            displacement_index = self.displacement_indices[i]

            if self.upsample_vector is None:
                template = self.templates_array_moved[displacement_index, unit_index, :, :]
            else:
                raise NotImplementedError

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
            traces[start_traces:end_traces] += wf.astype(self.dtype, copy=False)

        return traces.astype(self.dtype)

    def get_num_samples(self) -> int:
        return self.num_samples
