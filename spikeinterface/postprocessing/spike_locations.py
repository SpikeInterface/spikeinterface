import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs

from spikeinterface.core.template_tools import (get_template_extremum_channel,
                                                get_template_extremum_channel_peak_shift)

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


class SpikeLocationsCalculator(BaseWaveformExtractorExtension):
    """
    Computes spike locations from WaveformExtractor.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """    
    extension_name = 'spike_locations'
    
    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)

    def _set_params(self, ms_before=1., ms_after=1.5, method='center_of_mass',
                    method_kwargs={}):

        params = dict(ms_before=ms_before,
                      ms_after=ms_after,
                      method=method)
        params.update(**method_kwargs)
        return params        
    
    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.in1d(old_unit_ids, unit_ids))

        spike_mask = np.in1d(self.spikes['unit_ind'], unit_inds)
        new_spike_locations = self._extension_data['spike_locations'][spike_mask]
        return dict(spike_locations=new_spike_locations)
        
    def _run(self, **job_kwargs):
        """
        This function first transforms the sorting object into a `peaks` numpy array and then
        uses the`sortingcomponents.peak_localization.localize_peaks()` function to triangulate
        spike locations.
        """
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        job_kwargs = fix_job_kwargs(job_kwargs)

        we = self.waveform_extractor
        
        extremum_channel_inds = get_template_extremum_channel(we, outputs="index")
        self.spikes = we.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        
        spike_locations = localize_peaks(we.recording, self.spikes, **self._params, **job_kwargs)
        self._extension_data['spike_locations'] = spike_locations
    
    def get_data(self, outputs='concatenated'):
        """
        Get computed spike locations

        Parameters
        ----------
        outputs : str, optional
            'concatenated' or 'by_unit', by default 'concatenated'

        Returns
        -------
        spike_locations : np.array or dict
            The spike locations as a structured array (outputs='concatenated') or
            as a dict with units as key and spike locations as values.
        """
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        if outputs == 'concatenated':
            return self._extension_data['spike_locations']

        elif outputs == 'by_unit':
            locations_by_unit = []
            for segment_index in range(recording.get_num_segments()):
                i0 =np.searchsorted(self.spikes['segment_ind'], segment_index, side="left")
                i1 =np.searchsorted(self.spikes['segment_ind'], segment_index, side="right")
                spikes = self.spikes[i0: i1]
                locations = self._extension_data['spike_locations'][i0: i1]
                
                locations_by_unit.append({})
                for unit_ind, unit_id in enumerate(sorting.unit_ids):
                    mask = spikes['unit_ind'] == unit_ind
                    locations_by_unit[segment_index][unit_id] = locations[mask]
            return locations_by_unit

    @staticmethod
    def get_extension_function():
        return compute_spike_locations


WaveformExtractor.register_extension(SpikeLocationsCalculator)


def compute_spike_locations(waveform_extractor, load_if_exists=False, 
                            ms_before=1., ms_after=1.5, 
                            method='center_of_mass',
                            method_kwargs={},
                            outputs='concatenated',
                            **job_kwargs):
    """
    Localize spikes in 2D or 3D with several methods given the template.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        A waveform extractor object.
    load_if_exists : bool, optional, default: False
        Whether to load precomputed spike locations, if they already exist.
    ms_before : float
        The left window, before a peak, in milliseconds.
    ms_after : float
        The right window, after a peak, in milliseconds.
    method : str
        'center_of_mass' / 'monopolar_triangulation'
    method_kwargs : dict 
        Other kwargs depending on the method.
    outputs : str 
        'numpy' (default) / 'numpy_dtype' / 'dict'
    {}

    Returns
    -------
    spike_locations: np.array or list of dict
        The spike locations.
            - If 'concatenated' all locations for all spikes and all units are concatenated
            - If 'by_unit', locations are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(SpikeLocationsCalculator.extension_name):
        slc = waveform_extractor.load_extension(SpikeLocationsCalculator.extension_name)
    else:
        slc = SpikeLocationsCalculator(waveform_extractor)
        slc.set_params(ms_before=ms_before, ms_after=ms_after, method=method, method_kwargs=method_kwargs)
        slc.run(**job_kwargs)
    
    locs = slc.get_data(outputs=outputs)
    return locs


compute_spike_locations.__doc__.format(_shared_job_kwargs_doc)
