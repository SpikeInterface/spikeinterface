import numpy as np
import scipy.interpolate

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class RemoveArtifactsRecording(BasePreprocessor):
    """
    Removes stimulation artifacts from recording extractor traces. By default, 
    artifact periods are zeroed-out (mode = 'zeros'). This is only recommended 
    for traces that are centered around zero (e.g. through a prior highpass
    filter); if this is not the case, linear and cubic interpolation modes are
    also available, controlled by the 'mode' input argument.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to remove artifacts from
    list_triggers: list of list
        One list per segment of int with the stimulation trigger frames
    ms_before: float
        Time interval in ms to remove before the trigger events
    ms_after: float
        Time interval in ms to remove after the trigger events
    mode: str
        Determines what artifacts are replaced by. Can be one of the following:
            
        - 'zeros' (default): Artifacts are replaced by zeros.
        
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

    def __init__(self, recording, list_triggers, ms_before=0.5, ms_after=3.0, mode='zeros', fit_sample_spacing=1.):

        num_seg = recording.get_num_segments()
        if num_seg == 1 and isinstance(list_triggers, list) and np.isscalar(list_triggers[0]):
            # when unisque segment accept list instead of of list of list
            list_triggers = [list_triggers]

        # some check
        assert isinstance(list_triggers, list)
        assert len(list_triggers) == num_seg
        assert all(isinstance(list_triggers[i], list) for i in range(num_seg))
        assert mode in ('zeros', 'linear', 'cubic')

        sf = recording.get_sampling_frequency()
        pad = [int(ms_before * sf / 1000), int(ms_after * sf / 1000)]

        fit_sample_interval = int(fit_sample_spacing * sf / 1000.)
        fit_sample_range = fit_sample_interval * 2 + 1
        fit_samples = np.arange(0, fit_sample_range, fit_sample_interval)

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            triggers = list_triggers[seg_index]
            rec_segment = RemoveArtifactsRecordingSegment(parent_segment, triggers, pad, mode, fit_samples)
            self.add_recording_segment(rec_segment)

        list_triggers_int = [[int(trig) for trig in trig_seg] for trig_seg in list_triggers]
        self._kwargs = dict(recording=recording.to_dict(), list_triggers=list_triggers_int,
                            ms_before=float(ms_before), ms_after=float(ms_after), mode=mode,
                            fit_sample_spacing=fit_sample_spacing)


class RemoveArtifactsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, triggers, pad, mode, fit_samples):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.triggers = np.asarray(triggers, dtype='int64')
        self.pad = pad
        self.mode = mode
        self.fit_samples = fit_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        triggers = self.triggers[(self.triggers > start_frame) & (self.triggers < end_frame)] - start_frame

        pad = self.pad

        if self.mode == 'zeros':
            for trig in triggers:
                if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                    traces[trig - pad[0]:trig + pad[1], :] = 0
                elif trig - pad[0] <= 0 and trig + pad[1] >= end_frame - start_frame:
                    traces[:] = 0
                elif trig - pad[0] <= 0:
                    traces[:trig + pad[1], :] = 0
                elif trig + pad[1] >= end_frame - start_frame:
                    traces[trig - pad[0]:, :] = 0
        else:
            for trig in triggers:
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

        return traces


# function for API
def remove_artifacts(*args, **kwargs):
    return RemoveArtifactsRecording(*args, **kwargs)


remove_artifacts.__doc__ = RemoveArtifactsRecording.__doc__
