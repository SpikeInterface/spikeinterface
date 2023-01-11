"""Sorting components: peak waveform features."""
import numpy as np

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass, LocalizeMonopolarTriangulation
from spikeinterface.sortingcomponents.peak_pipeline import run_peak_pipeline, PeakPipelineStep



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
    job_kwargs = fix_job_kwargs(job_kwargs)

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
    def __init__(self, recording, ms_before=1., ms_after=1.,  peak_sign='neg', all_channels=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before, ms_after=ms_after)
        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channels:
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
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150., all_channels=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self.all_channels = all_channels
        self._kwargs = dict(all_channels=all_channels)
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channels:
            all_ptps = np.ptp(waveforms, axis=1)
        else:
            all_ptps = np.zeros(peaks.size)
            for main_chan in np.unique(peaks['channel_ind']):
                idx, = np.nonzero(peaks['channel_ind'] == main_chan)
                chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                all_ptps[idx] = np.max(np.ptp(wfs, axis=1))
        return all_ptps


class PeakToPeakLagsFeature(PeakPipelineStep):
    need_waveforms = True
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150., all_channels=True):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self.all_channels = all_channels
        self._kwargs = dict(all_channels=all_channels)
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channels:
            all_maxs = np.argmax(waveforms, axis=1)
            all_mins = np.argmin(waveforms, axis=1)
            all_lags = all_maxs - all_mins
        else:
            all_lags = np.zeros(peaks.size)
            for main_chan in np.unique(peaks['channel_ind']):
                idx, = np.nonzero(peaks['channel_ind'] == main_chan)
                chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                maxs = np.argmax(wfs, axis=1)
                mins = np.argmin(wfs, axis=1)
                lags = maxs - mins
                ptps = np.argmax(np.ptp(wfs, axis=1), axis=1)
                all_lags[idx] = lags[np.arange(len(idx)), ptps]
        return all_lags


class RandomProjectionsFeature(PeakPipelineStep):
    need_waveforms = True
    def __init__(self, recording, projections, ms_before=1., ms_after=1., local_radius_um=150., min_values=None):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self.projections = projections
        self.min_values = min_values
        self._kwargs = dict(projections=self.projections)
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        all_projections = np.zeros((peaks.size, self.projections.shape[1]), dtype=self._dtype)
        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            local_projections = self.projections[chan_inds, :]
            wf_ptp = (waveforms[idx][:, :, chan_inds]).ptp(axis=1)

            if self.min_values is not None:
                wf_ptp = (wf_ptp/self.min_values[chan_inds])**4

            denom = np.sum(wf_ptp, axis=1)
            mask = denom != 0

            all_projections[idx[mask]] = np.dot(wf_ptp[mask], local_projections)/(denom[mask][:, np.newaxis])
        return all_projections


class RandomProjectionsEnergyFeature(PeakPipelineStep):
    need_waveforms = True
    def __init__(self, recording, projections, ms_before=1., ms_after=1., local_radius_um=150., min_values=None):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self.projections = projections
        self.min_values = min_values
        self._kwargs = dict(projections=self.projections)
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        all_projections = np.zeros((peaks.size, self.projections.shape[1]), dtype=self._dtype)
        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            local_projections = self.projections[chan_inds, :]
            energies = np.linalg.norm(waveforms[idx][:, :, chan_inds], axis=1)

            if self.min_values is not None:
                energies = (energies/self.min_values[chan_inds])**4

            denom = np.sum(energies, axis=1)
            mask = denom != 0

            all_projections[idx[mask]] = np.dot(energies[mask], local_projections)/(denom[mask][:, np.newaxis])
        return all_projections


class StdPeakToPeakFeature(PeakToPeakFeature):
    need_waveforms = True
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150.):
        PeakToPeakFeature.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um, all_channels=False)

    def compute_buffer(self, traces, peaks, waveforms):
        all_ptps = np.zeros(peaks.size)
        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx][:, :, chan_inds]
            all_ptps[idx] = np.std(np.ptp(wfs, axis=1), axis=1)
        return all_ptps

class GlobalPeakToPeakFeature(PeakToPeakFeature):
    need_waveforms = True
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150.):
        PeakToPeakFeature.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um, all_channels=False)

    def compute_buffer(self, traces, peaks, waveforms):
        all_ptps = np.zeros(peaks.size)
        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx][:, :, chan_inds]
            all_ptps[idx] = np.max(wfs, axis=(1, 2)) - np.min(wfs, axis=(1, 2))
        return all_ptps

class KurtosisPeakToPeakFeature(PeakToPeakFeature):
    need_waveforms = True
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150.):
        PeakToPeakFeature.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um, all_channels=False)

    def compute_buffer(self, traces, peaks, waveforms):
        if self.all_channels:
            all_ptps = np.ptp(waveforms, axis=1)
        else:
            all_ptps = np.zeros(peaks.size)
            import scipy
            for main_chan in np.unique(peaks['channel_ind']):
                idx, = np.nonzero(peaks['channel_ind'] == main_chan)
                chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                all_ptps[idx] = scipy.stats.kurtosis(np.ptp(wfs, axis=1), axis=1)
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

            wfs = waveforms[idx][:, :, chan_inds]
            energy[idx] = np.linalg.norm(wfs, axis=(1, 2)) / chan_inds.size
        return energy


_features_class = {
    'amplitude': AmplitudeFeature,
    'ptp' : PeakToPeakFeature,
    'center_of_mass' : LocalizeCenterOfMass,
    'monopolar_triangulation' : LocalizeMonopolarTriangulation,
    'energy' : EnergyFeature,
    'std_ptp' : StdPeakToPeakFeature,
    'kurtosis_ptp' : KurtosisPeakToPeakFeature,
    'random_projections_ptp' : RandomProjectionsFeature,
    'random_projections_energy' : RandomProjectionsEnergyFeature,
    'ptp_lag' : PeakToPeakLagsFeature,
    'global_ptp' : GlobalPeakToPeakFeature
}