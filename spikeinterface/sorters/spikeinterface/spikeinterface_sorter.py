from spikeinterface.sorters import BaseSorter
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.toolkit import bandpass_filter, common_reference
from spikeinterface.core import load_extractor, BaseRecording
from spikeinterface.toolkit import get_noise_levels
from spikeinterface.core import extract_waveforms

class SpikeInterfaceSorter(BaseSorter):

    sorter_name = 'spikeinterface'

    _default_params = {
        'waveforms' : {'ms_before' : 2, 'ms_after' : 2, 'max_spikes_per_unit' : 200, 'overwrite' : True},
        'filtering' : {'method_kwargs' : {'dtype' : 'float32'}},
        'detection' : {'method' : 'locally_exclusive', 'method_kwargs' : {'peak_sign': 'neg', 'detect_threshold': 5, 'n_shifts' : 100}},
        'selection' : {'method' : 'smart_sampling_amplitudes', 'method_kwargs' : {'n_peaks' : 50000}},
        'localization' : {'method' : 'center_of_mass', 'method_kwargs' : {}},
        'clustering': {'method' : 'position', 'method_kwargs' : {}},
        'matching':  {'method' : 'circus-omp', 'method_kwargs' : {}},
        'common_reference': False,
        'job_kwargs' : {'n_jobs' : -1, 'chunk_duration' : '1s', 'verbose' : True}
    }

    @classmethod
    def is_installed(cls):
        return True

    @classmethod
    def get_sorter_version(cls):
        return "1.0"

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        recording = load_extractor(output_folder / 'spikeinterface_recording.json')
        sampling_rate = recording.get_sampling_frequency()

        filtering_params = params['filtering']['method_kwargs'].copy()
        recording_f = bandpass_filter(recording, **filtering_params)
        if params['common_reference']:
            recording_f = common_reference(recording_f)

        detection_params = params['detection']['method_kwargs'].copy()
        detection_params.update(params['job_kwargs'])

        peaks = detect_peaks(recording_f, method=params['detection']['method'], 
            **detection_params)

        selection_params = params['selection']['method_kwargs']
        noise_levels = get_noise_levels(recording_f, return_scaled=False)
        selection_params.update({'noise_levels' : noise_levels})

        selected_peaks = select_peaks(peaks, method=params['selection']['method'], **selection_params)

        localization_params = params['localization']['method_kwargs'].copy()
        localization_params.update(params['job_kwargs'])

        localizations = localize_peaks(recording_f, selected_peaks, method=params['localization']['method'], 
            **localization_params)

        labels, peak_labels = find_cluster_from_peaks(recording_f, selected_peaks, method=params['clustering']['method'], 
            method_kwargs=params['clustering']['method_kwargs'], **params['job_kwargs'])

        mask = peak_labels > -1
        sorting = NumpySorting.from_times_labels(selected_peaks['sample_ind'][mask], peak_labels[mask], sampling_rate)

        waveforms_params = params['waveforms'].copy()
        waveforms_params.update(params['job_kwargs'])
        we = extract_waveforms(recording_f, sorting, output_folder / "waveforms", **waveforms_params)

        matching_params = params['matching']['method_kwargs'].copy()
        matching_params['waveform_extractor'] = we
        matching_params.update({'noise_levels' : noise_levels})

        matching_job_params = params['job_kwargs'].copy()
        matching_job_params['chunk_duration'] = '100ms'

        spikes = find_spikes_from_templates(recording_f, method=params['matching']['method'], 
            method_kwargs=matching_params, **matching_job_params)

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