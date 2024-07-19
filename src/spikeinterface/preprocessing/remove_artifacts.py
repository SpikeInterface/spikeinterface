from __future__ import annotations

import warnings

import numpy as np

from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core import NumpySorting, estimate_templates


class RemoveArtifactsRecording(BasePreprocessor):
    """
    Removes stimulation artifacts from recording extractor traces. By default,
    artifact periods are zeroed-out (mode = "zeros"). This is only recommended
    for traces that are centered around zero (e.g. through a prior highpass
    filter); if this is not the case, linear and cubic interpolation modes are
    also available, controlled by the "mode" input argument.
    Note that several artifacts can be removed at once (potentially with
    distinct duration each), if labels are specified

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to remove artifacts from
    list_triggers : list of lists/arrays
        One list per segment of int with the stimulation trigger frames
    ms_before : float or None, default: 0.5
        Time interval in ms to remove before the trigger events.
        If None, then also ms_after must be None and a single sample is removed
    ms_after : float or None, default: 3.0
        Time interval in ms to remove after the trigger events.
        If None, then also ms_before must be None and a single sample is removed
    list_labels : list of lists/arrays or None
        One list per segment of labels with the stimulation labels for the given
        artifacts. labels should be strings, for JSON serialization.
        Required for "median" and "average" modes.
    mode : "zeros", "linear", "cubic", "average", "median", default: "zeros"
        Determines what artifacts are replaced by. Can be one of the following:

        - "zeros": Artifacts are replaced by zeros.

        - "median": The median over all artifacts is computed and subtracted for
            each occurence of an artifact

        - "average": The mean over all artifacts is computed and subtracted for each
            occurence of an artifact

        - "linear": Replacement are obtained through Linear interpolation between
           the trace before and after the artifact.
           If the trace starts or ends with an artifact period, the gap is filled
           with the closest available value before or after the artifact.

        - "cubic": Cubic spline interpolation between the trace before and after
           the artifact, referenced to evenly spaced fit points before and after
           the artifact. This is an option thatcan be helpful if there are
           significant LFP effects around the time of the artifact, but visual
           inspection of fit behaviour with your chosen settings is recommended.
           The spacing of fit points is controlled by "fit_sample_spacing", with
           greater spacing between points leading to a fit that is less sensitive
           to high frequency fluctuations but at the cost of a less smooth
           continuation of the trace.
           If the trace starts or ends with an artifact, the gap is filled with
           the closest available value before or after the artifact.
    fit_sample_spacing : float, default: 1.0
        Determines the spacing (in ms) of reference points for the cubic spline
        fit if mode = "cubic". Note : The actual fit samples are
        the median of the 5 data points around the time of each sample point to
        avoid excessive influence from hyper-local fluctuations.
    artifacts : dict or None, default: None
        If provided (when mode is "median" or "average") then it must be a dict with
        keys that are the labels of the artifacts, and values the artifacts themselves,
        on all channels (and thus bypassing ms_before and ms_after)
    sparsity : dict or None, default: None
        If provided (when mode is "median" or "average") then it must be a dict with
        keys that are the labels of the artifacts, and values that are boolean mask of
        the channels where the artifacts should be considered (for subtraction/scaling)
    scale_amplitude : False, default: False
        If true, then for mode "median" or "average" the amplitude of the template
        will be scaled in amplitude at each time occurence to minimize residuals
    time_jitter : float, default: 0
        If non 0, then for mode "median" or "average", a time jitter in ms
        can be allowed to minimize the residuals
    waveforms_kwargs : None
        Deprecated and ignored

    Returns
    -------
    removed_recording : RemoveArtifactsRecording
        The recording extractor after artifact removal
    """

    def __init__(
        self,
        recording,
        list_triggers,
        ms_before=0.5,
        ms_after=3.0,
        mode="zeros",
        fit_sample_spacing=1.0,
        list_labels=None,
        artifacts=None,
        sparsity=None,
        scale_amplitude=False,
        time_jitter=0,
        waveforms_kwargs=None,
    ):
        if waveforms_kwargs is not None:
            warnings("remove_artifacts() waveforms_kwargs is deprecated and ignored")

        available_modes = ("zeros", "linear", "cubic", "average", "median")
        num_seg = recording.get_num_segments()

        if num_seg == 1:
            if isinstance(list_triggers, (list, np.ndarray)) and np.isscalar(list_triggers[0]):
                # when unique segment accept list instead of of list of list/arrays
                list_triggers = [list_triggers]
            if isinstance(list_labels, (list, np.ndarray)) and np.isscalar(list_labels[0]):
                # when unique segment accept list instead of of list of list/arrays
                list_labels = [list_labels]

        # if no labels are given, assume single label
        if list_labels is None:
            list_labels = [[0] * len(i) for i in list_triggers]

        # some checks
        assert isinstance(list_triggers, list), "'list_triggers' must be a list (one per segment)"
        assert len(list_triggers) == num_seg, "'list_triggers' must have the same length as the number of segments"
        assert all(
            isinstance(list_triggers[i], (list, np.ndarray)) for i in range(num_seg)
        ), "Each element of 'list_triggers' must be array-like"

        if list_labels is not None:
            assert isinstance(list_labels, list), "'list_labels' must be a list (one per segment)"
            assert len(list_labels) == num_seg
            assert all(isinstance(list_labels[i], (list, np.ndarray)) for i in range(num_seg))

        assert mode in available_modes, f"mode {mode} is not an available mode: {available_modes}"

        if ms_before is None:
            assert ms_after is None, "To remove a single sample, set both ms_before and ms_after to None"
        else:
            ms_before = float(ms_before)
            ms_after = float(ms_after)

        fs = recording.get_sampling_frequency()
        if ms_before is not None:
            pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]
        else:
            pad = None

        fit_sample_interval = int(fit_sample_spacing * fs / 1000.0)
        fit_sample_range = fit_sample_interval * 2 + 1
        fit_samples = np.arange(0, fit_sample_range, fit_sample_interval)

        if mode in ["median", "average"]:
            assert time_jitter >= 0, "time jitter should be a positive value"
            time_pad = int(time_jitter * fs / 1000.0)

            if artifacts is not None:
                labels = []
                for sub_list in list_labels:
                    labels += list(np.unique(sub_list))
                for l in np.unique(labels):
                    assert l in artifacts.keys(), f"Artefacts are provided but label {l} has no value!"
            else:
                assert (
                    ms_before is not None and ms_after is not None
                ), f"ms_before/after should not be None for mode {mode}"
                sorting = NumpySorting.from_times_labels(list_triggers, list_labels, recording.get_sampling_frequency())

                nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
                nafter = int(ms_after * recording.sampling_frequency / 1000.0)

                templates = estimate_templates(
                    recording=recording,
                    spikes=sorting.to_spike_vector(),
                    unit_ids=sorting.unit_ids,
                    nbefore=nbefore,
                    nafter=nafter,
                    operator=mode,
                    return_scaled=False,
                )
                artifacts = {}
                for i, label in enumerate(sorting.unit_ids):
                    artifacts[label] = templates[i, :, :]

            if sparsity is not None:
                labels = []
                for sub_list in list_labels:
                    labels += list(np.unique(sub_list))
                for l in np.unique(labels):
                    assert l in sparsity.keys(), f"Sparsities are provided but label {l} has no value!"
        else:
            artifacts = None
            time_pad = None

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            triggers = list_triggers[seg_index]
            labels = list_labels[seg_index]
            rec_segment = RemoveArtifactsRecordingSegment(
                parent_segment, triggers, pad, mode, fit_samples, artifacts, labels, scale_amplitude, time_pad, sparsity
            )
            self.add_recording_segment(rec_segment)

        list_triggers_ = [[int(trig) for trig in trig_seg] for trig_seg in list_triggers]
        if list_labels is not None:
            list_labels_ = [list(lab_seg) for lab_seg in list_labels]
        else:
            list_labels_ = None
        self._kwargs = dict(
            recording=recording,
            list_triggers=list_triggers_,
            ms_before=ms_before,
            ms_after=ms_after,
            mode=mode,
            fit_sample_spacing=fit_sample_spacing,
            artifacts=artifacts,
            list_labels=list_labels_,
            scale_amplitude=scale_amplitude,
            time_jitter=time_jitter,
            sparsity=sparsity,
        )


class RemoveArtifactsRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        triggers,
        pad,
        mode,
        fit_samples,
        artifacts,
        labels,
        scale_amplitude,
        time_pad,
        sparsity,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.triggers = np.asarray(triggers, dtype="int64")
        self.pad = pad
        self.mode = mode
        self.artifacts = artifacts
        if self.artifacts is not None:
            for key, value in self.artifacts.items():
                self.artifacts[key] = np.array(value)
        self.labels = np.asarray(labels)
        self.fit_samples = fit_samples
        self.scale_amplitude = scale_amplitude
        self.time_pad = time_pad
        self.sparsity = sparsity

    def get_traces(self, start_frame, end_frame, channel_indices):
        if self.mode in ["average", "median"]:
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        else:
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()

        mask = (self.triggers >= start_frame) & (self.triggers < end_frame)
        triggers = self.triggers[mask] - start_frame
        labels = self.labels[mask]

        pad = self.pad

        if self.mode == "zeros":
            for trig in triggers:
                if pad is None:
                    traces[trig, :] = 0
                else:
                    if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                        traces[trig - pad[0] : trig + pad[1] + 1, :] = 0
                    elif trig - pad[0] <= 0 and trig + pad[1] >= end_frame - start_frame:
                        traces[:] = 0
                    elif trig - pad[0] <= 0:
                        traces[: trig + pad[1], :] = 0
                    elif trig + pad[1] >= end_frame - start_frame:
                        traces[trig - pad[0] :, :] = 0
        elif self.mode in ["linear", "cubic"]:
            import scipy.interpolate

            for trig in triggers:
                if pad is None:
                    pre_data_end_idx = trig - 1
                    post_data_start_idx = trig + 1
                else:
                    pre_data_end_idx = trig - pad[0] - 1
                    post_data_start_idx = trig + pad[1] + 1

                # Generate fit points from the sample points determined
                # pre_idx = pre_data_end_idx - self.rev_fit_samples + 1
                pre_idx = pre_data_end_idx - self.fit_samples[::-1]
                post_idx = post_data_start_idx + self.fit_samples

                # Get indices of the gap to fill
                gap_idx = np.arange(pre_data_end_idx + 1, post_data_start_idx + 0)

                # Make sure we are not going out of bounds
                gap_idx = gap_idx[gap_idx >= 0]
                gap_idx = gap_idx[gap_idx < traces.shape[0]]

                # correct for out of bounds indices on both sides:
                if np.max(post_idx) >= traces.shape[0]:
                    post_idx = post_idx[post_idx < traces.shape[0]]

                if np.min(pre_idx) < 0:
                    pre_idx = pre_idx[pre_idx >= 0]

                # fit x values
                all_idx = np.hstack((pre_idx, post_idx))

                # fit y values
                interp_traces = traces[all_idx, :]

                # Get the median value from 5 samples around each fit point
                # for robustness to noise / small fluctuations
                pre_vals = []  #  np.zeros((0, traces.shape[1]), dtype=traces.dtype)1
                for idx in iter(pre_idx):
                    if idx == pre_idx[-1]:
                        idxs = np.arange(idx - 3, idx + 1)
                    else:
                        idxs = np.arange(idx - 2, idx + 3)
                    if np.min(idxs) < 0:
                        idxs = idxs[idxs >= 0]
                    median_vals = np.median(traces[idxs, :], axis=0, keepdims=True)
                    pre_vals.append(median_vals)
                post_vals = []
                for idx in iter(post_idx):
                    if idx == post_idx[0]:
                        idxs = np.arange(idx, idx + 4)
                    else:
                        idxs = np.arange(idx - 2, idx + 3)
                    if np.max(idxs) >= traces.shape[0]:
                        idxs = idxs[idxs < traces.shape[0]]
                    median_vals = np.median(traces[idxs, :], axis=0, keepdims=True)
                    post_vals.append(median_vals)

                if len(all_idx) > 0:
                    interp_traces = np.concatenate(pre_vals + post_vals, axis=0)

                if self.mode == "cubic" and len(all_idx) >= 5:
                    # Enough fit points present on either side to do cubic spline fit:
                    interp_function = scipy.interpolate.interp1d(
                        all_idx, interp_traces, kind="cubic", axis=0, bounds_error=False, fill_value="extrapolate"
                    )
                    traces[gap_idx, :] = interp_function(gap_idx)
                elif self.mode == "linear" and len(all_idx) >= 2:
                    # Enough fit points present for a linear fit
                    interp_function = scipy.interpolate.interp1d(
                        all_idx, interp_traces, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate"
                    )
                    traces[gap_idx, :] = interp_function(gap_idx)
                elif len(pre_idx) > len(post_idx):
                    # not enough fit points, fill with nearest neighbour on side with the most data points
                    traces[gap_idx, :] = np.repeat(traces[[pre_idx[-1]], :], len(gap_idx), axis=0)
                elif len(post_idx) > len(pre_idx):
                    # not enough fit points, fill with nearest neighbour on side with the most data points
                    traces[gap_idx, :] = np.repeat(traces[[post_idx[0]], :], len(gap_idx), axis=0)
                elif len(all_idx) > 0:
                    # not enough fit points, both sides tied for most data points, fill with last pre value
                    traces[gap_idx, :] = np.repeat(traces[[pre_idx[-1]], :], len(gap_idx), axis=0)
                else:
                    # No data to interpolate from on either side of gap;
                    # Fill with zeros
                    traces[gap_idx, :] = 0

        elif self.mode in ["average", "median"]:
            for label, trig in zip(labels, triggers):
                if self.sparsity is not None:
                    mask = self.sparsity[label]
                else:
                    mask = None
                artifact_duration = len(self.artifacts[label])
                if self.time_pad > 0:
                    jitters = np.arange(-self.time_pad, self.time_pad, 1)
                else:
                    jitters = np.array([0])

                nb_jitters = len(jitters)
                best_amplitudes = np.zeros(nb_jitters, dtype=np.float32)

                for count, padding in enumerate(jitters):
                    t_trig = trig + padding

                    if t_trig - pad[0] >= 0 and t_trig + pad[1] < end_frame - start_frame:
                        trace_slice = slice(t_trig - pad[0], t_trig + pad[1])
                        artifact_slice = slice(0, artifact_duration)
                    elif t_trig - pad[0] < 0:
                        trace_slice = slice(0, t_trig + pad[1])
                        duration = t_trig + pad[1]
                        artifact_slice = slice(artifact_duration - duration, artifact_duration)
                    elif t_trig + pad[1] >= end_frame - start_frame:
                        trace_slice = slice(t_trig - pad[0], end_frame - start_frame)
                        duration = (end_frame - start_frame) - (t_trig - pad[0])
                        artifact_slice = slice(0, duration)

                    trace_slice_values = traces[trace_slice]
                    if mask is not None:
                        trace_slice_values = trace_slice_values[:, mask]

                    artifact_slice_values = self.artifacts[label][artifact_slice]

                    norm = np.linalg.norm(trace_slice_values) * np.linalg.norm(artifact_slice_values)
                    best_amplitudes[count] = (
                        np.dot(trace_slice_values.flatten(), artifact_slice_values.flatten()) / norm
                    )

                if nb_jitters > 0:
                    idx_best_jitter = np.argmax(best_amplitudes)
                    t_trig = trig + jitters[idx_best_jitter]

                    if t_trig - pad[0] >= 0 and t_trig + pad[1] < end_frame - start_frame:
                        trace_slice = slice(t_trig - pad[0], t_trig + pad[1])
                        artifact_slice = slice(0, artifact_duration)
                    elif t_trig - pad[0] < 0:
                        trace_slice = slice(0, t_trig + pad[1])
                        duration = t_trig + pad[1]
                        artifact_slice = slice(artifact_duration - duration, artifact_duration)
                    elif t_trig + pad[1] >= end_frame - start_frame:
                        trace_slice = slice(t_trig - pad[0], end_frame - start_frame)
                        duration = (end_frame - start_frame) - (t_trig - pad[0])
                        artifact_slice = slice(0, duration)
                else:
                    idx_best_jitter = 0

                if self.scale_amplitude:
                    best_amp = best_amplitudes[idx_best_jitter]
                else:
                    best_amp = 1

                if mask is not None:
                    traces[trace_slice][:, mask] -= (best_amp * self.artifacts[label][artifact_slice]).astype(
                        traces.dtype
                    )
                else:
                    traces[trace_slice] -= (best_amp * self.artifacts[label][artifact_slice]).astype(traces.dtype)
            traces = traces[:, channel_indices]

        return traces


# function for API
remove_artifacts = define_function_from_class(source_class=RemoveArtifactsRecording, name="remove_artifacts")
