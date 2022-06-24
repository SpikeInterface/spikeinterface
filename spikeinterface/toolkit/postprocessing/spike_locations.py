import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

from .template_tools import (get_template_extremum_channel,
                             get_template_extremum_channel_peak_shift)


class SpikeLocationsCalculator(BaseWaveformExtractorExtension):
    """
    Localize spikes in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe (width)
       * Y is axis 1 of the probe (depth)
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
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
    {}

    Returns
    -------
    spike_locations: np.array
        The spike locations.
            - If 'concatenated' all locations for all spikes and all units are concatenated
    """    
    extension_name = 'spike_locations'
    
    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)
        
        self.locations = None
        self.spikes = None

    def _set_params(self, ms_before=1., ms_after=1.5, method='center_of_mass',
                    method_kwargs={}):

        params = dict(ms_before=ms_before,
                      ms_after=ms_after,
                      method=method,
                      method_kwargs=method_kwargs)
        return params        
        
    def _specific_load_from_folder(self):
        we = self.waveform_extractor

        extremum_channel_inds = get_template_extremum_channel(we, outputs="index")
        self.spikes = we.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        self.locations =np.load( self.extension_folder / 'spike_locations.npy')

    def _reset(self):
        self.locations = None
    
    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.in1d(old_unit_ids, unit_ids))

        spike_mask = np.in1d(self.spikes['unit_ind'], unit_inds)
        new_location = self.locations[spike_mask]
        np.save(new_waveforms_folder / 'spike_locations.npy', new_location)
        
    def compute_locations(self, **job_kwargs):
        """
        This function first transforms the sorting object into a `peaks` numpy array and then
        uses the`sortingcomponents.peak_localization.localize_peaks()` function to triangulate
        spike locations.
        """
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        
        we = self.waveform_extractor
        
        extremum_channel_inds = get_template_extremum_channel(we, outputs="index")
        self.spikes = we.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        
        self.locations = localize_peaks(we.recording, self.spikes, **self._params, **job_kwargs)
        np.save(self.extension_folder / 'spike_locations.npy', self.locations)

    
    def get_locations(self, outputs='concatenated'):
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        if outputs == 'concatenated':
            return self.locations

        elif outputs == 'by_unit':
            locations_by_unit = []
            for segment_index in range(recording.get_num_segments()):
                i0 =np.searchsorted(self.spikes['segment_ind'], segment_index, side="left")
                i1 =np.searchsorted(self.spikes['segment_ind'], segment_index, side="right")
                spikes = self.spikes[i0: i1]
                locations = self.locations[i0: i1]
                
                locations_by_unit.append({})
                for unit_ind, unit_id in enumerate(sorting.unit_ids):
                    mask = spikes['unit_ind'] == unit_ind
                    locations_by_unit[segment_index][unit_id] = locations[mask]
            return locations_by_unit


SpikeLocationsCalculator.__doc__.format(_shared_job_kwargs_doc)

WaveformExtractor.register_extension(SpikeLocationsCalculator)


def compute_spike_locations(waveform_extractor, load_if_exists=False, 
                            ms_before=1., ms_after=1.5, 
                            method='center_of_mass',
                            method_kwargs={},
                            outputs='concatenated',
                            **job_kwargs):


    folder = waveform_extractor.folder
    ext_folder = folder / SpikeLocationsCalculator.extension_name

    if load_if_exists and ext_folder.is_dir():
        slc = SpikeLocationsCalculator.load_from_folder(folder)
    else:
        slc = SpikeLocationsCalculator(waveform_extractor)
        slc.set_params(ms_before=ms_before, ms_after=ms_after, method=method, method_kwargs=method_kwargs)
        slc.compute_locations(**job_kwargs)
    
    locs = slc.get_locations(outputs=outputs)
    return locs


compute_spike_locations.__doc__ = SpikeLocationsCalculator.__doc__
