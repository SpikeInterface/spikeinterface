from .si_based import ComponentsBasedSorter

from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels, extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore

import numpy as np

class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = 'tridesclous2'

    _default_params = {
        'apply_preprocessing': True,
    
        'general' : {'ms_before' : 2.5, 'ms_after' : 3.5, 'local_radius_um' : 100},
        
        'filtering' : {'freq_min' : 300, 'freq_max': 8000.},
        'detection' : {'peak_sign': 'neg', 'detect_threshold': 5, 'exclude_sweep_ms': 0.4},
        
        'hdbscan_kwargs' : {"min_cluster_size" : 25,  "allow_single_cluster" : True, "core_dist_n_jobs" : -1, "cluster_selection_method" : "leaf"},
        
        'waveforms' : { 'max_spikes_per_unit' : 300},
        'selection' : {'n_peaks_per_channel' : 5000, 'min_n_peaks' : 20000},
        'localization' : {'max_distance_um':1000, 'optimizer': 'minimize_with_log_penality'},
        'matching':  {'peak_shift_ms':  0.2,},
        'job_kwargs' : {'n_jobs' : -1, 'chunk_duration' : '1s', 'progress_bar': True}
    }

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params['job_kwargs'].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs['progress_bar'] = verbose
    
        # this is importanted only on demand because numba import are too heavy
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        
        import hdbscan

        recording_raw = load_extractor(sorter_output_folder.parent / 'spikeinterface_recording.json')
        
        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        
        # preprocessing
        if params['apply_preprocessing']:
            recording = bandpass_filter(recording_raw, **params['filtering'])
            recording = common_reference(recording)
            recording = zscore(recording, dtype='float32')
            noise_levels = np.ones(num_chans, dtype='float32')
        else:
            recording = recording_raw
            noise_levels = get_noise_levels(recording, return_scaled=False)
        
        
        # detection
        detection_params = params['detection'].copy()
        detection_params['local_radius_um'] = params['general']['local_radius_um']
        detection_params['noise_levels'] = noise_levels
        peaks = detect_peaks(recording, method='locally_exclusive',  **detection_params, **job_kwargs)

        if verbose:
            print('We found %d peaks in total' %len(peaks))
        
        # selection
        selection_params = params['selection'].copy()
        selection_params['n_peaks'] = params['selection']['n_peaks_per_channel'] * num_chans
        selection_params['n_peaks'] = max(selection_params['min_n_peaks'], selection_params['n_peaks'])
        selection_params['noise_levels'] = noise_levels
        some_peaks = select_peaks(peaks, method='smart_sampling_amplitudes', select_per_channel=False,
                                  **selection_params)

        if verbose:
            print('We kept %d peaks for clustering' %len(some_peaks))

        # localization
        localization_params = params['localization'].copy()
        localization_params['local_radius_um'] = params['general']['local_radius_um']
        peak_locations = localize_peaks(recording, some_peaks, method='monopolar_triangulation',
                                        **localization_params, **job_kwargs)
        
        #~ print(peak_locations.dtype)
        
        # features = localisations only
        peak_features = np.zeros((peak_locations.size, 3), dtype='float64')
        for i, dim in enumerate(['x', 'y', 'z']):
            peak_features[:, i] = peak_locations[dim]
        
        # clusering is hdbscan
        
        
        out = hdbscan.hdbscan(peak_features, **params['hdbscan_kwargs'])
        peak_labels = out[0]
        
        mask = peak_labels >= 0
        labels = np.unique(peak_labels[mask])
        
        # extract waveform for template matching
        sorting_temp = NumpySorting.from_times_labels(some_peaks['sample_ind'][mask], peak_labels[mask],
                                                      sampling_frequency)
        sorting_temp = sorting_temp.save(folder=sorter_output_folder / 'sorting_temp')
        waveforms_params = params['waveforms'].copy()
        waveforms_params['ms_before'] = params['general']['ms_before']
        waveforms_params['ms_after'] = params['general']['ms_after']
        we = extract_waveforms(recording, sorting_temp, sorter_output_folder / "waveforms_temp",
                               **waveforms_params, **job_kwargs)
        
        ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces
        matching_params = params['matching'].copy()
        matching_params['waveform_extractor'] = we
        matching_params['noise_levels'] = noise_levels
        matching_params['peak_sign'] = params['detection']['peak_sign']
        matching_params['detect_threshold'] = params['detection']['detect_threshold']
        matching_params['local_radius_um'] = params['general']['local_radius_um']

        # TODO: route that params
        #~ 'num_closest' : 5,
        #~ 'sample_shift': 3,
        #~ 'ms_before': 0.8,
        #~ 'ms_after': 1.2,
        #~ 'num_peeler_loop':  2,
        #~ 'num_template_try' : 1,
        
        spikes = find_spikes_from_templates(recording, method='tridesclous',  method_kwargs=matching_params,
                                            **job_kwargs)

        if verbose:
            print('We found %d spikes' %len(spikes))

        sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], sampling_frequency)
        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting

