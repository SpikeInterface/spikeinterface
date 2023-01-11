"""Sorting components: peak localization."""
import numpy as np
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, split_job_kwargs, fix_job_kwargs

from .peak_pipeline import run_peak_pipeline, PeakPipelineStep
from .tools import make_multi_method_doc

from ..postprocessing.unit_localization import (dtype_localize_by_method,
                                                possible_localization_methods,
                                                solve_monopolar_triangulation,
                                                make_radial_order_parents,
                                                enforce_decrease_shells_ptp)


def localize_peaks(recording, peaks, method='center_of_mass', **kwargs):
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.

    {method_doc}

    {job_doc}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ('x', 'y') or ('x', 'y', 'z', 'alpha').
    """
    assert method in possible_localization_methods, f"Method {method} is not supported. Choose from {possible_localization_methods}"

    method_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if method == 'center_of_mass':
        step = LocalizeCenterOfMass(recording, **method_kwargs)
    elif method == 'monopolar_triangulation':
        step = LocalizeMonopolarTriangulation(recording, **method_kwargs)
    elif method == "peak_channel":
        step = LocalizePeakChannel(recording,  **method_kwargs)
    
    peak_locations = run_peak_pipeline(recording, peaks, [step], job_kwargs, job_name='localize peaks',
                                       squeeze_output=True)
    
    return peak_locations


class LocalizeBase(PeakPipelineStep):
    def __init__(self, recording, ms_before, ms_after, local_radius_um):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)

    def get_dtype(self):
        return self._dtype


class LocalizePeakChannel(PeakPipelineStep):
    """Localize peaks using the center of mass method."""
    
    need_waveforms = False
    name = 'peak_channel'
    params_doc = """
    """

    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self._dtype = np.dtype(dtype_localize_by_method['center_of_mass'])

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks):  # Why buffer?
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for index, main_chan in enumerate(peaks['channel_ind']):
            locations = self.contact_locations[main_chan, :]
            peak_locations['x'][index] = locations[0]
            peak_locations['y'][index] = locations[1]

        return peak_locations


class LocalizeCenterOfMass(PeakPipelineStep):
    """Localize peaks using the center of mass method."""
    need_waveforms = True
    name = 'center_of_mass'
    params_doc = """
    ms_before: float
        Time in ms to cut before spike peak
    ms_after: float
        Time in ms to cut after spike peak
    local_radius_um: float
        Radius in um for channel sparsity.
    """
    def __init__(self, recording, ms_before=1., ms_after=1., local_radius_um=150):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self._dtype = np.dtype(dtype_localize_by_method['center_of_mass'])

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks, waveforms):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for main_chan in np.unique(peaks['channel_ind']):
            idx, = np.nonzero(peaks['channel_ind'] == main_chan)
            chan_inds, = np.nonzero(self.neighbours_mask[main_chan])
            local_contact_locations = self.contact_locations[chan_inds, :]

            wf_ptp = (waveforms[idx][:, :, chan_inds]).ptp(axis=1)
            coms = np.dot(wf_ptp, local_contact_locations)/(np.sum(wf_ptp, axis=1)[:,np.newaxis])
            peak_locations['x'][idx] = coms[:, 0]
            peak_locations['y'][idx] = coms[:, 1]

        return peak_locations


class LocalizeMonopolarTriangulation(PeakPipelineStep):
    """Localize peaks using the monopolar triangulation method.

    Notes
    -----
    This method is from  Julien Boussard, Erdem Varol and Charlie Windolf
    See spikeinterface.postprocessing.unit_localization.
    """
    need_waveforms = False
    name = 'monopolar_triangulation'
    params_doc = """
    ms_before: float
        Time in ms to cut before spike peak
    ms_after: float
        Time in ms to cut after spike peak
    local_radius_um: float
        For channel sparsity.
    max_distance_um: float, default: 1000
        Boundary for distance estimation.
    enforce_decrese : None or "radial"
        If+how to enforce spatial decreasingness for PTP vectors.
    """
    def __init__(self, recording, 
                        ms_before=1., ms_after=1.,
                        local_radius_um=150,
                        max_distance_um=1000,
                        optimizer='minimize_with_log_penality',
                        enforce_decrease=False):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)
        self._kwargs.update(dict(max_distance_um=max_distance_um,
                                 optimizer=optimizer,
                                 enforce_decrease=enforce_decrease))

        self.max_distance_um = max_distance_um
        self.optimizer = optimizer

        if enforce_decrease:
            self.enforce_decrease_radial_parents = make_radial_order_parents(self.contact_locations, self.neighbours_mask)
        else:
            self.enforce_decrease_radial_parents = None

        self._dtype = np.dtype(dtype_localize_by_method['monopolar_triangulation'])

    def get_dtype(self):
        return self._dtype

    def compute_buffer(self, traces, peaks):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for i, peak in enumerate(peaks):
            sample_ind = peak['sample_ind']
            chan_mask = self.neighbours_mask[peak['channel_ind'], :]
            chan_inds = np.flatnonzero(chan_mask)
            local_contact_locations = self.contact_locations[chan_inds, :]

            # wf is (nsample, nchan) - chan is only neighbor
            wf = traces[sample_ind - self.nbefore:sample_ind + self.nafter, :][:, chan_inds]

            wf_ptp = wf.ptp(axis=0)
            if self.enforce_decrease_radial_parents is not None:
                enforce_decrease_shells_ptp(
                    wf_ptp, peak['channel_ind'], self.enforce_decrease_radial_parents, in_place=True
                )

            peak_locations[i] = solve_monopolar_triangulation(wf_ptp, local_contact_locations,
                                                              self.max_distance_um, self.optimizer)

        return peak_locations


# LocalizePeakChannel is not include in doc because it is not a good idea to use it
_methods_list = [LocalizeCenterOfMass, LocalizeMonopolarTriangulation]
localize_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
localize_peaks.__doc__ = localize_peaks.__doc__.format(
                                    method_doc=method_doc,
                                    job_doc=_shared_job_kwargs_doc)
