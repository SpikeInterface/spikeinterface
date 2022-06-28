from spikeinterface.sorters import BaseSorter
from .peak_detection import detect_peaks
from .peak_localization import localize_peaks
from .peak_detection.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting

class SpikeInterfaceSorter(BaseSorter):

    sorter_name = 'spikeinterface'

    _default_params = {
        'filtering' : {'method' : 'bandpass_filter'},
        'detection' : {'method' : 'locally_exclusive', 'method_kwargs' : {}},
        'selection' : {'method' : 'smart_sampling_amplitudes', 'method_kwargs' : {'n_peaks' : 50000}}
        'localization' : {'method' : 'center_of_mass', 'method_kwargs' : {}},
        'clustering': {'method' : 'position_and_features', 'method_kwargs' : {}},
        'matching':  {'method' : 'circus-omp', 'method_kwargs' : {}},
        'peak_sign': 'neg',
        'detect_threshold': 5,
        'common_reference': True
    }

    def run(self, raise_error=True):

        peaks = detect_peaks(self.recording_f, method=self.params['detection']['method'], 
            **self.params['detection']['method_kwargs'])

        selected_peaks = select_peaks(peaks, method=self.params['selection']['method'], 
            **self.params['selection']['method_kwargs'])

        localizations = localize_peaks(self.recording_f, selected_peaks, method=self.params['localization']['method'], 
            **self.params['localization']['method_kwargs'])

        labels, peak_labels = find_cluster_from_peaks(self.recording_f, selected_peaks, method=self.params['clustering']['method'], 
            method_kwargs=self.params['clustering']['method_kwargs'])

        spikes = find_spikes_from_templates(self.recording_f, method=self.params['matching']['method'], 
            method_kwargs=self.params['matching']['method_kwargs'], **self.job_kwargs)

        sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], self.sampling_rate)
        
        return sorting

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        self.recording = recording
        self.recording_f = bandpass_filter(self.recording, dtype='float32')
        if self.params['common_reference']:
            self.recording_f = common_reference(self.recording_f)

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        sorting = NumpySorting(folder_path=output_folder)
        return sorting