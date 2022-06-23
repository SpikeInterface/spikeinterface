import numpy as np
import shutil

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

from .template_tools import (get_template_extremum_channel,
                             get_template_extremum_channel_peak_shift)


class SpikeLocationsCalculator(BaseWaveformExtractorExtension):
    """
    Localize spikes in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
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

        self._locations = None
        self._all_spikes = None

    def _set_params(self, ms_before=1, ms_after=1, method='center_of_mass',
                    method_kwargs={}):

        params = dict(ms_before=ms_before,
                      ms_after=ms_after,
                      method=method,
                      method_kwargs=method_kwargs)
        return params        
        
    def _specific_load_from_folder(self):
        recording = self.waveform_extractor.recording
        sorting = self.waveform_extractor.sorting

        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        self._all_spikes = all_spikes

        self._locations = []
        for segment_index in range(recording.get_num_segments()):
            file_locs = self.extension_folder / f'locations_segment_{segment_index}.npy'
            locs_seg = np.load(file_locs)
            self._locations.append(locs_seg)

    def _reset(self):
        self._locations = None
    
    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # load filter and save amplitude files
        for seg_index in range(self.waveform_extractor.recording.get_num_segments()):
            loc_file_name = f"locations_segment_{seg_index}.npy"
            locs = np.load(self.extension_folder / loc_file_name)
            _, all_labels = self.waveform_extractor.sorting.get_all_spike_trains()[seg_index]
            filtered_idxs = np.in1d(all_labels, np.array(unit_ids)).nonzero()
            np.save(new_waveforms_folder / self.extension_name /
                    loc_file_name, locs[filtered_idxs])
    
        
    def compute_locations(self, **job_kwargs):
        """
        This function first transforms the sorting object into a `peaks` numpy array and then
        uses the`sortingcomponents.peak_localization.localize_peaks()` function to triangulate
        spike locations.
        """
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        self._all_spikes = all_spikes
        
        extremum_channel_inds = get_template_extremum_channel(we, outputs="index")
        sorting_peaks = we.sorting.to_spike_vector()
        unit_inds = sorting_peaks['unit_ind']
        channel_inds = [extremum_channel_inds[u_id] for u_id in we.sorting.unit_ids[unit_inds]] 
        sorting_peaks['unit_ind'] = channel_inds
        dtype_names = list(sorting_peaks.dtype.names)
        unit_ind_index = dtype_names.index("unit_ind")
        dtype_names[unit_ind_index] = "channel_ind"
        sorting_peaks.dtype.names = tuple(dtype_names)
        
        localization_params = {**self._params, **job_kwargs}
        
        locs = localize_peaks(recording, sorting_peaks, 
                              **localization_params)

        self._locations = []
        for segment_index in range(recording.get_num_segments()):
            mask = sorting_peaks["segment_ind"] == segment_index
            locs_seg = locs[mask]
            self._locations.append(locs)
            
            # save to folder
            file_locs = self.extension_folder / f'locations_segment_{segment_index}.npy'
            np.save(file_locs, locs_seg)
    
    def get_locations(self, outputs='concatenated'):
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        if outputs == 'concatenated':
            return self._locations

        elif outputs == 'by_unit':
            locations_by_unit = []
            for segment_index in range(recording.get_num_segments()):
                locations_by_unit.append({})
                for unit_index, unit_id in enumerate(sorting.unit_ids):
                    spike_times, spike_labels = self._all_spikes[segment_index]
                    mask = spike_labels == unit_index
                    amps = self._locations[segment_index][mask]
                    locations_by_unit[segment_index][unit_id] = amps
            return locations_by_unit


SpikeLocationsCalculator.__doc__.format(_shared_job_kwargs_doc)

WaveformExtractor.register_extension(SpikeLocationsCalculator)


def compute_spike_locations(waveform_extractor, load_if_exists=False, 
                            ms_before=1, ms_after=1, 
                            method='center_of_mass',
                            method_kwargs={},
                            outputs='concatenated',
                            **job_kwargs):


    folder = waveform_extractor.folder
    ext_folder = folder / SpikeLocationsCalculator.extension_name

    if load_if_exists and ext_folder.is_dir():
        sac = SpikeLocationsCalculator.load_from_folder(folder)
    else:
        sac = SpikeLocationsCalculator(waveform_extractor)
        sac.set_params(ms_before=ms_before, ms_after=ms_after, method=method, method_kwargs=method_kwargs)
        sac.compute_locations(**job_kwargs)
    
    locs = sac.get_locations(outputs=outputs)
    return locs


compute_spike_locations.__doc__ = SpikeLocationsCalculator.__doc__
