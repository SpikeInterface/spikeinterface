"""Sorting components: peak waveform features."""
import numpy as np

from spikeinterface.core import get_chunk_with_margin, get_channel_distances

from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep

from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass


def compute_features_from_peaks(
    recording,
    peaks,
    feature_list=["ptp", ],
    feature_params={},
    ms_before=1.,
    ms_after=1.,
    **job_kwargs,
):
    """Extract features on the fly from the recording given a list of peaks. 

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    feature_list: List of features to be computed.
            - amplitude
            - ptp
            - com
            - energy
    ms_before: float
        The duration in ms before the peak for extracting the features (default 1 ms)
    ms_after: float
        The duration in ms  after the peakfor extracting the features (default 1 ms)

    {}

    Returns
    -------
    A tuple of features. Even if there is one feature.
    Every feature have shape[0] == peaks.shape[0].
    dtype and other dim depends on features.

    """

    steps = []
    for feature_name in feature_list:
        Class = _features_class[feature_name]
        params = feature_params.get(feature_name, {}).copy()
        if Class.need_waveforms:
            params.update(dict(ms_before=ms_before, ms_after=ms_after))
        step = Class(recording, **params)
        steps.append(step)

    features = run_peak_pipeline(
        recording, peaks, steps, job_kwargs, job_name='features_from_peaks', squeeze_output=False)

    return features


class AmplitudeFeature(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channel=True):
        PeakPipelineStep.__init__(
            self, recording, ms_before=ms_before, ms_after=ms_after)
        self.all_channel = all_channel
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channel=all_channel, peak_sign=peak_sign))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channel:
            if self.peak_sign == 'neg':
                amplitudes = np.min(waveforms, axis=1)
            elif self.peak_sign == 'pos':
                amplitudes = np.max(waveforms, axis=1)
            elif self.peak_sign == 'both':
                amplitudes = np.max(np.abs(waveforms, axis=1))
        else:
            if self.peak_sign == 'neg':
                amplitudes = np.min(waveforms, axis=(1, 2))
            elif self.peak_sign == 'pos':
                amplitudes = np.max(waveforms, axis=(1, 2))
            elif self.peak_sign == 'both':
                amplitudes = np.max(np.abs(waveforms), axis=(1, 2))
        return amplitudes


class PeakToPeakFeature(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150., all_channel=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self.all_channel = all_channel
        self._kwargs = dict(all_channel=all_channel)
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channel:
            all_ptps = np.ptp(waveforms, axis=1)
        else:
            all_ptps = np.zeros(peaks.shape[0])

            for main_chan in np.unique(peaks['channel_ind']):
                idx, = np.nonzero(peaks['channel_ind'] == main_chan)
                chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                all_ptps[idx] = np.max(np.ptp(wfs, axis=1))
        return all_ptps


class EnergyFeature(PeakPipelineStep):
    need_waveforms = True

    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=50.):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)

    def get_dtype(self):
        return np.dtype('float32')

    def compute_buffer(self, traces, peaks, waveforms):
        energy = np.zeros(peaks.size, dtype='float32')
        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx]
            energy[idx] = np.linalg.norm(
                wfs[:, :, chan_inds], axis=(1, 2)) / chan_inds.size
        return energy


_features_class = {
    'amplitude': AmplitudeFeature,
    'ptp': PeakToPeakFeature,
    'com': LocalizeCenterOfMass,
    'energy': EnergyFeature,

}

# @pierre this is for you because this features is not usefull
# TODO : 'dist_com_vs_max_ptp_channel'
