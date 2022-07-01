from spikeinterface.sorters import BaseSorter
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels
from spikeinterface.core import extract_waveforms

class Spykingcircus2Sorter(BaseSorter):

    sorter_name = 'spykingcircus2'

    _default_params = {
        'general' : {'ms_before' : 2.5, 'ms_after' : 3.5, 'local_radius_um' : 50},
        'waveforms' : { 'max_spikes_per_unit' : 200, 'overwrite' : True},
        'filtering' : {'freq_min' : 300, 'freq_max' : 6000, 'dtype' : 'float32'},
        'detection' : {'peak_sign': 'neg', 'detect_threshold': 5, 'exclude_sweep_ms' : 2},
        'selection' : {'n_peaks_per_channel' : 2000, 'min_n_peaks' : 20000},
        'localization' : {},
        'clustering': {},
        'matching':  {},
        'registration' : {},
        'common_reference': True,
        'job_kwargs' : {'n_jobs' : -1, 'chunk_duration' : '1s', 'verbose' : True}
    }

    @classmethod
    def is_installed(cls):
        return True 

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        recording = load_extractor(output_folder / 'spikeinterface_recording.json')
        sampling_rate = recording.get_sampling_frequency()

        ## First, we are filtering the data
        filtering_params = params['filtering'].copy()
        recording_f = bandpass_filter(recording, **filtering_params)

        if params['common_reference']:
            recording_f = common_reference(recording_f)


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
        selection_params['n_peaks'] = params['selection']['n_peaks_per_channel'] * recording.get_num_channels()
        selection_params['n_peaks'] = max(selection_params['min_n_peaks'], selection_params['n_peaks'])

        noise_levels = get_noise_levels(recording_f, return_scaled=False)
        selection_params.update({'noise_levels' : noise_levels})
        selected_peaks = select_peaks(peaks, method='smart_sampling_amplitudes', select_per_channel=False, **selection_params)

        if verbose:
            print('We kept %d peaks for clustering' %len(selected_peaks))

        ## We localize the CoM of the peaks
        localization_params = params['localization'].copy()
        if 'local_radius_um' not in localization_params:
            localization_params['local_radius_um'] = params['general']['local_radius_um']

        localizations = localize_peaks(recording_f, selected_peaks, method='center_of_mass', 
            method_kwargs=localization_params, **params['job_kwargs'])

        ## We launch a clustering (using hdbscan) relying on positions and features extracted on
        ## the fly from the snippets
        clustering_params = params['clustering'].copy()
        clustering_params.update(params['waveforms'])
        clustering_params['peak_locations'] = localizations
        clustering_params['job_kwargs'] = params['job_kwargs']

        if 'local_radius_um' not in clustering_params:
            clustering_params['local_radius_um'] = params['general']['local_radius_um']

        labels, peak_labels = find_cluster_from_peaks(recording_f, selected_peaks, method='position_and_features', 
            method_kwargs=clustering_params)

        ## We get the labels for our peaks
        mask = peak_labels > -1
        sorting = NumpySorting.from_times_labels(selected_peaks['sample_ind'][mask], peak_labels[mask], sampling_rate)

        ## We get the templates our of such a clustering
        waveforms_params = params['waveforms'].copy()
        waveforms_params.update(params['job_kwargs'])
        we = extract_waveforms(recording_f, sorting, output_folder / "waveforms", **waveforms_params)

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
        sorting = sorting.save(folder=output_folder / "sorting")

        return sorting

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        params['job_kwargs']['verbose'] = verbose
        params['job_kwargs']['progress_bar'] = verbose
        cls.set_params_to_folder(recording, output_folder, params, verbose)

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        sorting = load_extractor(output_folder / "sorting")
        return sorting