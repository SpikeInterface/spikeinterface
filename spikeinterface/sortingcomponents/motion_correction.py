import numpy as np
import scipy.interpolate
import sklearn

from tqdm import tqdm

import sklearn.metrics

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def correct_motion_on_peaks(peaks, peak_locations, times,
        motion, temporal_bins, spatial_bins,
        direction='y', progress_bar=False):
    """
    Given the output of estimate_motion() apply inverse motion on peak location.

    Parameters
    ----------
    peaks: np.array
        peaks vector
    peak_locations: 
        peaks location vector
    times: 
        times vector of recording
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used 

    Returns
    -------
    corrected_peak_locations: np.array
        Motion-corrected peak locations
    """
    corrected_peak_locations = peak_locations.copy()

    if spatial_bins is None:
        # rigid motion interpolation 1D
        sample_bins = np.searchsorted(times, temporal_bins)
        f = scipy.interpolate.interp1d(sample_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        shift = f(peaks['sample_ind'])
        corrected_peak_locations[direction] -= shift
    else:
        # non rigid motion = interpolation 2D
        sample_bins = np.searchsorted(times, temporal_bins)
        f = scipy.interpolate.RegularGridInterpolator((sample_bins, spatial_bins), motion, 
                                                      method='linear', bounds_error=False, fill_value=None)
        shift = f(list(zip(peaks['sample_ind'], peak_locations[direction])))
        corrected_peak_locations[direction] -= shift

    return corrected_peak_locations



def channel_motions_over_time(times, channel_locations, motion, temporal_bins, spatial_bins, direction=1):
    """
    Interpolate the channel motion over time given motion matrix.

    Parameters
    ----------
    times: np.array 1d
        Times vector
    channel_location: np.array 2d
        Channel location with shape (n, 2) or (n, 3)
    motion: np.array 2D
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion
                        equal temporal_bins.shape[0] when "none rigid"
    temporal_bins: np.array
        Temporal bins in second.
    spatial_bins: None or np.array
        Bins for non-rigid motion. If None, rigid motion is used 
    direction: int in (0, 1, 2)
        Dimension of shift in channel_locations.
    Returns
    -------
    channel_motions: np.array
        Shift over time by channel
        Shape (times.shape[0], channel_location.shape[0])
    """
    
    num_chans = channel_locations.shape[0]
    num_samples = times.shape[0]

    # clip to times
    l0 = max(0, np.searchsorted(temporal_bins, times[0], side='left') - 1)
    l1 = np.searchsorted(temporal_bins, times[-1], side='right') + 1 

    temporal_bins = temporal_bins[l0:l1]
    motion = motion[l0:l1, :]

    if spatial_bins is None:
        # rigid motion interpolation 1D
        f = scipy.interpolate.interp1d(temporal_bins, motion[:, 0], bounds_error=False, fill_value="extrapolate")
        channel_motions = f(times)
        channel_motions.reshape(-1, 1)
    else:
        # non rigid motion interpolation 2D

        # (1) inperpolate in time
        f = scipy.interpolate.interp1d(temporal_bins, motion, kind='linear', 
                                       axis=0, bounds_error=False, fill_value="extrapolate")
        motion_high = f(times)

        # (2) inperpolate on space
        f = scipy.interpolate.interp1d(spatial_bins, motion_high, kind='linear', 
                                       axis=1, bounds_error=False, fill_value="extrapolate")

        locs = channel_locations[:, direction]
        channel_motions = f(locs)

    return channel_motions




def correct_motion_on_traces(traces, times, channel_locations, motion, temporal_bins, spatial_bins, direction=1,):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------

    Returns
    -------

    """
    assert HAVE_NUMBA 
    assert times.shape[0] == traces.shape[0]

    num_samples = times.shape[0]

    traces_corrected = np.zeros_like(traces)
    # print(traces_corrected.shape)

    if spatial_bins is None:
        # rigid motion interpolation 1D
        raise NotImplementedError
    else:
        # non rigid motion = interpolation 2D
        
        # regroup times by closet temporal_bins
        bin_inds = _get_closest_ind(temporal_bins, times)

        # inperpolation kernel will be the same per temporal bin        
        for bin_ind in np.unique(bin_inds):
            mask = bin_ind == bin_inds
            # print('bin_ind', bin_ind, np.sum(mask))

            # Step 1 : interpolation channel motion for this temporal bin
            f = scipy.interpolate.interp1d(spatial_bins, motion[bin_ind, :], kind='linear', 
                                        axis=0, bounds_error=False, fill_value="extrapolate")
            locs = channel_locations[:, direction]
            channel_motions = f(locs)
            channel_locations_moved = channel_locations.copy()
            channel_locations_moved[:, direction] += channel_motions


            # Step 2 : interpolate trace
            # interpolation is done with Inverse Distance Weighted
            # because it is simple to implement
            # Instead vwe should use use the convex hull, Delaunay triangulation http://www.qhull.org/
            # scipy.interpolate.LinearNDInterpolator and qhull.Delaunay should help for this
            distances = sklearn.metrics.pairwise_distances(channel_locations_moved, channel_locations, metric='euclidean')
            num_chans = channel_locations.shape[0]
            num_closest = 3
            closest_chans = np.zeros((num_chans, num_closest), dtype='int64')
            weights = np.zeros((num_chans, num_closest), dtype='float32')
            for c in range(num_chans):
                ind_sorted = np.argsort(distances[c, ])
                closest_chans[c, :] = ind_sorted[:num_closest]
                dists = distances[c, ind_sorted[:num_closest]]
                if dists[0] == 0.:
                    # no interpolation the first have zeros distance
                    weights[c, :] = 0
                    weights[c, 0] = 1
                else:
                    # Inverse Distance Weighted
                    w = 1 / dists
                    w /= np.sum(w)
                    weights[c, :] = w
            my_inverse_weighted_distance_interpolation(traces, traces_corrected, closest_chans, weights)
        
    return traces_corrected



if HAVE_NUMBA:
    @numba.jit(parallel=False)
    def my_inverse_weighted_distance_interpolation(traces, traces_corrected, closest_chans, weights):
        num_sample = traces.shape[0] 
        num_chan = traces.shape[1]
        num_closest = closest_chans.shape[1]
        for sample_ind in range(num_sample):
            for chan_ind in range(num_chan):
                v = 0
                for i in range(num_closest):
                    other_chan = closest_chans[chan_ind, i]
                    v +=  weights[chan_ind, i] * traces[sample_ind, other_chan]
                traces_corrected[sample_ind, chan_ind] = v



def _get_closest_ind(array, values):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs




def correct_motion_on_traces_OLD(traces, times, channel_locations, motion, temporal_bins, spatial_bins, direction=1,):
    """
    Apply inverse motion with spatial interpolation on traces.

    Traces can be full traces, but also waveforms snippets.

    Parameters
    ----------

    Returns
    -------

    """
    
    assert times.shape[0] == traces.shape[0]

    num_samples = times.shape[0]

    traces_corrected = np.zeros_like(traces)
    # print(traces_corrected.shape)

    if spatial_bins is None:
        # rigid motion interpolation 1D
        raise NotImplementedError
    else:
        # non rigid motion = interpolation 2D
        
        channel_motions = channel_motions_over_time(times, channel_locations, motion,
                                                     temporal_bins, spatial_bins, direction=direction)

        # print(num_samples)
        for i in tqdm(range(num_samples)):
        # for i in tqdm(range(50000)):
            channel_locations_moved = channel_locations.copy()
            channel_locations_moved[:, direction] += channel_motions[i, :]

            v = scipy.interpolate.griddata(channel_locations_moved, traces[i, :],
                                                                (channel_locations),
                                                                 method='linear',
                                                                 # method='nearest',
                                                                 fill_value=np.nan,
                                                                 )
            traces_corrected[i, :] = v

            # traces_corrected[i, :] = scipy.interpolate.griddata(channel_locations_moved, traces[i, :],
            #                                                     (channel_locations),
            #                                                      method='linear')

            # f = scipy.interpolate.interp2d(channel_locations_moved[:, 0],
            #                                channel_locations_moved[:, 1],
            #                                traces[i, :], kind='linear',
            #                                bounds_error=False, fill_value=np.nan
            #                                )
            # v = f(channel_locations[:, 0], channel_locations[:, 1])
            # print(v.shape, channel_locations[:, 0].shape, channel_locations_moved[:, 0].shape)
            # print(v)
            # traces_corrected[i, :] = f(channel_locations[:, 0], channel_locations[:, 1])
            
        
    return traces_corrected




from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class CorrectMotionRecording(BasePreprocessor):
    """

    Parameters
    ----------

    Returns
    -------
    """
    name = 'correct_motion'

    def __init__(self, recording, motion, temporal_bins, spatial_bins, direction=1):

        BasePreprocessor.__init__(self, recording)

        channel_locations = recording.get_channel_locations()

        for parent_segment in recording._recording_segments:
            rec_segment = CorrectMotionRecordingSegment(parent_segment, channel_locations, 
                                                motion, temporal_bins, spatial_bins, direction)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), motion=motion, temporal_bins=temporal_bins,
                            spatial_bins=spatial_bins, direction=direction)
        # self.is_dumpable= False

class CorrectMotionRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, channel_locations, motion, temporal_bins, spatial_bins, direction):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.channel_locations = channel_locations
        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.direction = direction
        

    def get_traces(self, start_frame, end_frame, channel_indices):
        

        if self.time_vector is not None:
            times = np.asarray(self.time_vector[start_frame:end_frame])
        else:
            times = np.arange(end_frame - start_frame, dtype='float64')
            times /= self.sampling_frequency
            t0 = start_frame / self.sampling_frequency
            if self.t_start is not None:
                t0 = t0 + self.t_start
            times += t0


        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices=None)

        # print(traces.shape, times.shape, self.channel_locations, self.motion, self.temporal_bins, self.spatial_bins)
        trace2 = correct_motion_on_traces(traces, times, self.channel_locations, self.motion,
                                 self.temporal_bins, self.spatial_bins, direction=self.direction)

        if trace2 is not None:
            trace2 = trace2[:, channel_indices]
        
        return trace2
