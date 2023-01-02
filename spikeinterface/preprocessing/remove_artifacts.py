import numpy as np
import scipy.interpolate

from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core import NumpySorting, extract_waveforms

class RemoveArtifactsRecording(BasePreprocessor):
    """
    Removes stimulation artifacts from recording extractor traces. By default, 
    artifact periods are zeroed-out (mode = 'zeros'). This is only recommended 
    for traces that are centered around zero (e.g. through a prior highpass
    filter); if this is not the case, linear and cubic interpolation modes are
    also available, controlled by the 'mode' input argument.
    Note that several artefacts can be removed at once (potentially with 
    distinct duration each), if labels are specified

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to remove artifacts from
    list_triggers: list of list
        One list per segment of int with the stimulation trigger frames
    ms_before: float or None
        Time interval in ms to remove before the trigger events.
        Either `ms_before` or `frames_before` (see below) can be given, but not both:
            if both `ms_before` and `frames_before` are given, throws an error;
            if both `ms_before` and `frames_before` are None, then
                (i) `ms_after` and `frames_after` must also be None (otherwise throws an error) and
                (ii) a single sample is removed
    ms_after: float or None
        Time interval in ms to remove after the trigger events.
        Either `ms_after` or `frames_after` (see below) can be given, but not both:
            if both `ms_after` and `frames_after` are given, throws an error;
            if both `ms_after` and `frames_after` are None, then
                (i) `ms_before` and `frames_before` must also be None (otherwise throws an error) and
                (ii) a single sample is removed
    frames_before : int or None (default: None)
        Time interval in frames to remove before the trigger events.
        Use this instead of `ms_before` for discontinuous recordings (e.g. those with missing samples).
        See comments for `ms_before` above for how it interacts with other params.
    frames_after : int or None (default: None)
        Time interval in frames to remove after the trigger events.
        Use this instead of `ms_after` for discontinuous recordings (e.g. those with missing samples).
        See comments for `ms_after` above for how it interacts with other params.
    list_labels: list of list or None
        One list per segment of labels with the stimulation labels for the given
        artefacs
    mode: str
        Determines what artifacts are replaced by. Can be one of the following:
            
        - 'zeros' (default): Artifacts are replaced by zeros.

        - 'median': The median over all artefacts is computed and subtracted for 
            each occurence of an artefact

        - 'average': The mean over all artefacts is computed and subtracted for each 
            occurence of an artefact
        
        - 'linear': Replacement are obtained through Linear interpolation between
           the trace before and after the artifact.
           If the trace starts or ends with an artifact period, the gap is filled
           with the closest available value before or after the artifact.
        
        - 'cubic': Cubic spline interpolation between the trace before and after
           the artifact, referenced to evenly spaced fit points before and after
           the artifact. This is an option thatcan be helpful if there are
           significant LFP effects around the time of the artifact, but visual
           inspection of fit behaviour with your chosen settings is recommended.
           The spacing of fit points is controlled by 'fit_sample_spacing', with
           greater spacing between points leading to a fit that is less sensitive
           to high frequency fluctuations but at the cost of a less smooth
           continuation of the trace.
           If the trace starts or ends with an artifact, the gap is filled with
           the closest available value before or after the artifact.
    fit_sample_spacing: float
        Determines the spacing (in ms) of reference points for the cubic spline
        fit if mode = 'cubic'. Default = 1ms. Note: The actual fit samples are 
        the median of the 5 data points around the time of each sample point to
        avoid excessive influence from hyper-local fluctuations.
        

    Returns
    -------
    removed_recording: RemoveArtifactsRecording
        The recording extractor after artifact removal    
    """
    name = 'remove_artifacts'

    def __init__(self, recording, list_triggers, ms_before=0.5, ms_after=3.0, 
                 frames_before=None, frames_after=None,
                 mode='zeros', fit_sample_spacing=1., list_labels=None):

        num_seg = recording.get_num_segments()
        if num_seg == 1 and isinstance(list_triggers, list) and np.isscalar(list_triggers[0]):
            # when unisque segment accept list instead of of list of list
            list_triggers = [list_triggers]

        if num_seg == 1 and isinstance(list_labels, list):
            if not isinstance(list_labels[0], list):
                # when unisque segment accept list instead of of list of list
                list_labels = [list_labels]

        if list_labels is None:
            list_labels = [[0]*len(i) for i in list_triggers]

        # some check
        if list_labels is not None:
            assert isinstance(list_labels, list)
            assert len(list_labels) == num_seg
            assert all(isinstance(list_labels[i], list) for i in range(num_seg))

        assert isinstance(list_triggers, list)
        assert len(list_triggers) == num_seg
        assert all(isinstance(list_triggers[i], list) for i in range(num_seg))
        assert mode in ('zeros', 'linear', 'cubic', 'average', 'median')

        if (ms_before is None) and (frames_before is None):
            if (ms_after is not None) or (frames_after is not None):
                raise ValueError("You must specify either `ms_before` and `frames_before`",
                                 " (unless you want to remove single samples, in which case",
                                 " set `ms_after` and `frames_after` to None)")
            else:
                print("Only processing samples in `list_trigger`")
        elif (ms_before is not None) and (frames_before is None):
            ms_before = float(ms_before)
        elif (ms_before is None) and (frames_before is not None):
            frames_before = int(frames_before)
        else:
            raise ValueError("You cannot specify both `ms_before` and `frames_before`; choose one.")
        
        if (ms_after is None) and (frames_after is None):
            if (ms_before is not None) or (frames_before is not None):
                raise ValueError("You must specify either `ms_after` and `frames_after`",
                                 " (unless you want to remove single samples, in which case",
                                 " set `ms_before` and `frames_before` to None)")
        elif (ms_after is not None) and (frames_after is None):
            ms_after = float(ms_after)
        elif (ms_after is None) and (frames_after is not None):
            frames_after = int(frames_after)
        else:
            raise ValueError("You cannot specify both `ms_after` and `frames_after`; choose one.")
        
        sf = recording.get_sampling_frequency()
        if ms_before is not None:
            if ms_after is not None:
                pad = [int(ms_before * sf / 1000), int(ms_after * sf / 1000)]
            else:
                pad = [int(ms_before * sf / 1000), frames_after]
        else:
            if ms_after is not None:
                pad = [frames_before, int(ms_after * sf / 1000)]
            else:
                if (frames_before is not None) and (frames_after is not None):
                    pad = [frames_before, frames_after]
                else:
                    pad = []

        fit_sample_interval = int(fit_sample_spacing * sf / 1000.)
        fit_sample_range = fit_sample_interval * 2 + 1
        fit_samples = np.arange(0, fit_sample_range, fit_sample_interval)

        # how to pass in `frames_before`` and `frames_after` to `extract_waveforms`?
        if mode in ['median', 'average']:
            sorting = NumpySorting.from_times_labels(list_triggers, list_labels, recording.get_sampling_frequency())
            waveforms_params = {'ms_before' : ms_before, 'ms_after' : ms_after}
            w = extract_waveforms(recording, sorting, None, mode='memory', **waveforms_params, return_scaled=False)
            artefacts = {}
            for label in w.sorting.unit_ids:
                artefacts[label] = w.get_template(label, mode=mode)
        else:
            artefacts = None

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            triggers = list_triggers[seg_index]
            labels = list_labels[seg_index]
            rec_segment = RemoveArtifactsRecordingSegment(parent_segment, triggers, pad, mode, fit_samples, artefacts, labels)
            self.add_recording_segment(rec_segment)

        list_triggers_int = [[int(trig) for trig in trig_seg] for trig_seg in list_triggers]
        self._kwargs = dict(recording=recording.to_dict(), list_triggers=list_triggers_int,
                            ms_before=ms_before, ms_after=ms_after, frames_before=frames_before, frames_after=frames_after,
                            mode=mode, fit_sample_spacing=fit_sample_spacing, artefacts=artefacts, list_labels=list_labels)


class RemoveArtifactsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, triggers, pad, mode, fit_samples, artefacts, labels):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.triggers = np.asarray(triggers, dtype='int64')
        self.pad = pad
        self.mode = mode
        self.artefacts = artefacts
        self.labels = np.asarray(labels)
        self.fit_samples = fit_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        mask = (self.triggers >= start_frame) & (self.triggers < end_frame)
        triggers = self.triggers[mask] - start_frame
        labels = self.labels[mask]

        pad = self.pad

        if self.mode == 'zeros':
            for trig in triggers:
                if pad is None:
                    traces[trig, :] = 0
                else:
                    if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                        traces[trig - pad[0]:trig + pad[1] + 1, :] = 0
                    elif trig - pad[0] <= 0 and trig + pad[1] >= end_frame - start_frame:
                        traces[:] = 0
                    elif trig - pad[0] <= 0:
                        traces[:trig + pad[1], :] = 0
                    elif trig + pad[1] >= end_frame - start_frame:
                        traces[trig - pad[0]:, :] = 0
        elif self.mode in ['linear', 'cubic']:
            for trig in triggers:
                if pad is None:
                    pre_data_end_idx = trig - 1
                    post_data_start_idx = trig + 1
                else:
                    pre_data_end_idx = trig - pad[0] - 1
                    post_data_start_idx = trig + pad[1] + 1

                # Generate fit points from the sample points determined
                #  pre_idx = pre_data_end_idx - self.rev_fit_samples + 1
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
                    if np.min(idx) < 0:
                        idx = idx[idx >= 0]
                    median_vals = np.median(traces[idxs, :], axis=0, keepdims=True)
                    pre_vals.append(median_vals)
                post_vals = []
                for idx in iter(post_idx):
                    if idx == post_idx[0]:
                        idxs = np.arange(idx, idx + 4)
                    else:
                        idxs = np.arange(idx - 2, idx + 3)
                    if np.max(idx) >= traces.shape[0]:
                        idx = idx[idx < traces.shape[0]]
                    median_vals = np.median(traces[idxs, :], axis=0, keepdims=True)
                    post_vals.append(median_vals)

                if len(all_idx) > 0:
                    interp_traces = np.concatenate(pre_vals + post_vals, axis=0)

                if self.mode == 'cubic' and len(all_idx) >= 5:
                    # Enough fit points present on either side to do cubic spline fit:
                    interp_function = scipy.interpolate.interp1d(all_idx, interp_traces,
                                                                 kind='cubic', axis=0, bounds_error=False,
                                                                 fill_value='extrapolate')
                    traces[gap_idx, :] = interp_function(gap_idx)
                elif self.mode == 'linear' and len(all_idx) >= 2:
                    # Enough fit points present for a linear fit
                    interp_function = scipy.interpolate.interp1d(all_idx, interp_traces,
                                                                 kind='linear', axis=0, bounds_error=False,
                                                                 fill_value='extrapolate')
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
        elif self.mode in ['average', 'median']:
            for label, trig in zip(labels, triggers):
                if pad is None:
                    traces[trig, :] -= self.artefacts[label][trig, :]    
                else:
                    if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                        traces[trig-pad[0]:trig+pad[1], :] -= self.artefacts[label]
                    elif trig - pad[0] < 0:
                        duration = pad[1] + pad[0] - (pad[0] - trig)
                        traces[:trig+pad[1], :] -= self.artefacts[label][duration:]
                    elif trig + pad[1] >= end_frame - start_frame:
                        duration = (end_frame - start_frame) - (trig - pad[0])
                        traces[trig-pad[0]:, :] -= self.artefacts[label][:duration]        

        return traces


# function for API
remove_artifacts = define_function_from_class(source_class=RemoveArtifactsRecording, name="remove_artifacts")
