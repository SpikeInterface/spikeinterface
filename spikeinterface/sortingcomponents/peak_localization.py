"""Sorting components: peak localization."""

import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc


from ..postprocessing.unit_localization import (dtype_localize_by_method,
                                                possible_localization_methods,
                                                solve_monopolar_triangulation,
                                                make_radial_order_parents,
                                                enforce_decrease_shells_ptp)


from .peak_pipeline import run_peak_pipeline, PeakPipelineStep



def localize_peaks(recording, peaks, ms_before=1, ms_after=1, method='center_of_mass',
                   method_kwargs={}, **job_kwargs):
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
    ms_before: float
        The left window, before a peak, in milliseconds.
    ms_after: float
        The right window, after a peak, in milliseconds.
    method: 'center_of_mass' or 'monopolar_triangulation'
        Method to use.
    method_kwargs: dict of kwargs method
        Keyword arguments for the chosen method:
            'center_of_mass':
                * local_radius_um: float
                    For channel sparsity.
            'monopolar_triangulation':
                * local_radius_um: float
                    For channel sparsity.
                * max_distance_um: float, default: 1000
                    Boundary for distance estimation.
                * enforce_decrese : None or "radial"
                    If+how to enforce spatial decreasingness for PTP vectors.
    {}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ('x', 'y') or ('x', 'y', 'z', 'alpha').
    """
    assert method in possible_localization_methods, f"Method {method} is not supported. Choose from {possible_localization_methods}"

    if method == 'center_of_mass':
        step = LocalizeCenterOfMass(recording, ms_before=ms_before, ms_after=ms_after, **method_kwargs)
    elif method == 'monopolar_triangulation':
        step = LocalizeMonopolarTriangulation(recording, ms_before=ms_before, ms_after=ms_after, **method_kwargs)
        
    peak_locations = run_peak_pipeline(recording, peaks, [step], job_kwargs, job_name='localize peaks', squeeze_output=True)
    
    return peak_locations

localize_peaks.__doc__ = localize_peaks.__doc__.format(_shared_job_kwargs_doc)


class LocalizeBase(PeakPipelineStep):
    def __init__(self, recording, ms_before, ms_after, local_radius_um):
        PeakPipelineStep.__init__(self, recording, ms_before=ms_before,
                                  ms_after=ms_after, local_radius_um=local_radius_um)

    def get_dtype(self):
        return self._dtype


class LocalizeCenterOfMass(PeakPipelineStep):
    """Localize peaks using the center of mass method."""
    need_waveforms = True
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
            peak_locations[i] = solve_monopolar_triangulation(wf_ptp, local_contact_locations, self.max_distance_um, self.optimizer)

        return peak_locations

