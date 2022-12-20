from .si_based import ComponentsBasedSorter

import os
import shutil
import numpy as np
import os

from spikeinterface.core import (NumpySorting,  load_extractor, BaseRecording,
    get_noise_levels, extract_waveforms)
from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore

try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

class Spykingcircus2Sorter(ComponentsBasedSorter):

    sorter_name = 'spykingcircus2'

    _default_params = {
        'general' : {'ms_before' : 2, 'ms_after' : 2, 'local_radius_um' : 100},
        'waveforms' : {'max_spikes_per_unit' : 200, 'overwrite' : True},
        'filtering' : {'dtype' : 'float32'},
        'detection' : {'peak_sign': 'neg', 'detect_threshold': 5},
        'selection' : {'n_peaks_per_channel' : 5000, 'min_n_peaks' : 20000},
        'localization' : {},
        'clustering': {},
        'matching':  {},
        'registration' : {},
        'apply_preprocessing': True,
        'shared_memory' : False,
        'job_kwargs' : {'n_jobs' : -1, 'chunk_duration' : '1s', 'verbose' : False}
    }

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):

        assert HAVE_HDBSCAN, 'spykingcircus2 needs hdbscan to be installed'
    
        params['job_kwargs']['verbose'] = verbose
        params['job_kwargs']['progress_bar'] = verbose

        if "n_jobs" in params["job_kwargs"]:
            if params["job_kwargs"]["n_jobs"] == -1:
                params["job_kwargs"]["n_jobs"] = os.cpu_count()
    
        # this is importanted only on demand because numba import are too heavy
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

        recording = load_extractor(sorter_output_folder.parent / 'spikeinterface_recording.json')
        sampling_rate = recording.get_sampling_frequency()
        num_channels = recording.get_num_channels()

        ## First, we are filtering the data
        filtering_params = params['filtering'].copy()
        if params['apply_preprocessing']:
            #if recording.is_filtered == True:
            #    print('Looks like the recording is already filtered, check preprocessing!')
            recording_f = bandpass_filter(recording, **filtering_params)
            recording_f = common_reference(recording_f)
        else:
            recording_f = recording

        recording_f = zscore(recording_f, dtype='float32')

        ## Then, we are detecting peaks with a locally_exclusive method
        detection_params = params['detection'].copy()
        detection_params.update(params['job_kwargs'])
        if 'local_radius_um' not in detection_params:
            detection_params['local_radius_um'] = params['general']['local_radius_um']
        if 'exclude_sweep_ms' not in detection_params:
            detection_params['exclude_sweep_ms'] = max(params['general']['ms_before'], params['general']['ms_after'])

        peaks = detect_peaks(recording_f, method='locally_exclusive', 
                             **detection_params)

        if verbose:
            print('We found %d peaks in total' %len(peaks))

        ## We subselect a subset of all the peaks, by making the distributions os SNRs over all
        ## channels as flat as possible
        selection_params = params['selection']
        selection_params['n_peaks'] = params['selection']['n_peaks_per_channel'] * num_channels
        selection_params['n_peaks'] = max(selection_params['min_n_peaks'], selection_params['n_peaks'])

        noise_levels = np.ones(num_channels, dtype=np.float32)
        selection_params.update({'noise_levels' : noise_levels})
        selected_peaks = select_peaks(peaks, method='smart_sampling_amplitudes', select_per_channel=False, 
                                      **selection_params)

        if verbose:
            print('We kept %d peaks for clustering' %len(selected_peaks))

        ## We launch a clustering (using hdbscan) relying on positions and features extracted on
        ## the fly from the snippets
        clustering_params = params['clustering'].copy()
        clustering_params.update(params['waveforms'])
        clustering_params.update(params['general'])
        clustering_params.update(dict(shared_memory=params['shared_memory']))
        clustering_params['job_kwargs'] = params['job_kwargs']
        clustering_params['tmp_folder'] = sorter_output_folder / "clustering"

        labels, peak_labels = find_cluster_from_peaks(recording_f, selected_peaks, method='random_projections',
            method_kwargs=clustering_params)

        ## We get the labels for our peaks
        mask = peak_labels > -1
        sorting = NumpySorting.from_times_labels(selected_peaks['sample_ind'][mask], peak_labels[mask], sampling_rate)
        clustering_folder = sorter_output_folder / "clustering"
        if clustering_folder.exists():
            shutil.rmtree(clustering_folder)

        sorting = sorting.save(folder=clustering_folder)

        ## We get the templates our of such a clustering
        waveforms_params = params['waveforms'].copy()
        waveforms_params.update(params['job_kwargs'])

        if params['shared_memory']:
            mode = 'memory'
            waveforms_folder = None
        else:
            mode = 'folder'
            waveforms_folder = sorter_output_folder / "waveforms"

        we = extract_waveforms(recording_f, sorting, waveforms_folder, mode=mode, **waveforms_params, return_scaled=False)

        ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces
        matching_params = params['matching'].copy()
        matching_params['waveform_extractor'] = we
        matching_params.update({'noise_levels' : noise_levels})

        matching_job_params = params['job_kwargs'].copy()
        matching_job_params['chunk_duration'] = '100ms'

        spikes = find_spikes_from_templates(recording_f, method='circus-omp', 
            method_kwargs=matching_params, **matching_job_params)

        if verbose:
            print('We found %d spikes' %len(spikes))

        ## And this is it! We have a spyking circus
        sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], sampling_rate)
        sorting_folder = sorter_output_folder / "sorting"

        if sorting_folder.exists():
            shutil.rmtree(sorting_folder)

        sorting = sorting.save(folder=sorting_folder)

        return sorting

