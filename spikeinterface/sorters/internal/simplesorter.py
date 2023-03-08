from .si_based import ComponentsBasedSorter

from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels, extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore

import numpy as np


import pickle
import json


class SimpleSorter(ComponentsBasedSorter):
    sorter_name = 'simple'

    _default_params = {
        'apply_preprocessing': False,
    
        'general' : {'ms_before' : 1.0, 'ms_after' : 1.5, 'local_radius_um' : 120.},
        
        'filtering' : {'freq_min' : 300, 'freq_max': 8000.},
        'detection' : {'peak_sign': 'neg', 'detect_threshold': 6., 'exclude_sweep_ms': 0.4},
        
        'hdbscan_kwargs' : {"min_cluster_size" : 25,  "allow_single_cluster" : True,
                            "core_dist_n_jobs" : -1, "cluster_selection_method" : "leaf"},
        'job_kwargs' : {'n_jobs' : -1, 'chunk_duration' : '1s'}
    }

    @classmethod
    def get_sorter_version(cls):
        return "1.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params['job_kwargs'].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs['progress_bar'] = verbose
    
        # this is importanted only on demand because numba import are too heavy
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.postprocessing import compute_principal_components
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

        # extract ALL waveforms
        sorting_temp = NumpySorting.from_times_labels(peaks['sample_ind'], np.zeros(peaks.size, dtype='int64'), 
                                                    sampling_frequency)
        sorting_temp = sorting_temp.save(folder=sorter_output_folder / 'sorting_temp')
        waveforms_params = params['waveforms'].copy()
        waveforms_params['ms_before'] = params['general']['ms_before']
        waveforms_params['ms_after'] = params['general']['ms_after']
        waveforms_params['max_spikes_per_unit'] = None
        waveforms_params['sparse'] = False
        we = extract_waveforms(recording, sorting_temp, sorter_output_folder / "waveforms_temp",
                            **waveforms_params, **job_kwargs)

        # compute PCs
        pc = compute_principal_components(we, n_components=3, mode="by_channel_global", sparsity=None, whiten=True)
        pc_features = pc.get_projections(unit_id=0)
        pc_features_flat = pc_features.reshape(pc_features.shape[0], -1)

        # run hdscan for clustering
        out = hdbscan.hdbscan(pc_features_flat, **params['hdbscan_kwargs'])
        peak_labels = out[0]

        # keep positive labels
        keep = peak_labels >= 0
        sorting_final = NumpySorting.from_times_labels(peaks['sample_ind'][keep], peak_labels[keep], sampling_frequency)
        sorting_final = sorting_final.save(folder=sorter_output_folder / "sorting")

        return sorting_final

