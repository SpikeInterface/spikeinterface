
from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import plot_probe_map, plot_agreement_matrix, plot_comparison_collision_by_similarity, plot_unit_templates, plot_unit_waveforms
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.comparison.comparisontools import make_matching_events
from spikeinterface.postprocessing import get_template_extremum_channel, get_template_extremum_amplitude, compute_center_of_mass
from spikeinterface.core import get_noise_levels

import time
import string, random
import pylab as plt
import os
import numpy as np

class BenchmarkPeakLocalization:

    def __init__(self, recording, gt_sorting, exhaustive_gt=True, job_kwargs={}, tmp_folder=None, verbose=True):
        self.verbose = verbose
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.exhaustive_gt = exhaustive_gt
        self.recording_f = recording
        self.sampling_rate = self.recording_f.get_sampling_frequency()

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self.waveforms = {}
        self.pcas = {}
        self.templates = {}

    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_folder)


    @property
    def gt_positions(self):
        if self._gt_positions is None:
            self.localize_gt_peaks()
        return self._gt_positions

    def detect_peaks(self, method_kwargs={'method' : 'locally_exclusive'}):
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Detecting peaks with method {method}')
        self._peaks = detect_peaks(self.recording_f, **method_kwargs, **self.job_kwargs)

    def localize_peaks(self, method_kwargs = {'method' : 'center_of_mass'}):
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Localizing peaks with method {method}')
        self._positions = localize_peaks(self.recording_f, self.peaks, **method_kwargs, **self.job_kwargs)

    def localize_gt_peaks(self, method_kwargs = {'method' : 'center_of_mass'}):
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Localizing gt peaks with method {method}')
        self._gt_positions = localize_peaks(self.recording_f, self.gt_peaks, **method_kwargs, **self.job_kwargs)

    def run(self, peaks=None, positions=None, delta=0.2):
        t_start = time.time()
        

        if peaks is not None:
            self._peaks = peaks

        nb_peaks = len(self.peaks)

        if positions is not None:
            self._positions = positions

        times1 = self.gt_sorting.get_all_spike_trains()[0]
        times2 = self.peaks['sample_ind']

        print("The gt recording has {} peaks and {} have been detected".format(len(times1[0]), len(times2)))
        
        matches = make_matching_events(times1[0], times2, int(delta*self.sampling_rate/1000))
        self.matches = matches

        self.deltas = {'labels' : [], 'delta' : matches['delta_frame']}
        self.deltas['labels'] = times1[1][matches['index1']]

        #print(len(times1[0]), len(matches['index1']))
        gt_matches = matches['index1']
        self.sliced_gt_sorting = NumpySorting.from_times_labels(times1[0][gt_matches], times1[1][gt_matches], self.sampling_rate, unit_ids = self.gt_sorting.unit_ids)
        ratio = 100*len(gt_matches)/len(times1[0])
        print("Only {0:.2f}% of gt peaks are matched to detected peaks".format(ratio))

        matches = make_matching_events(times2, times1[0], int(delta*self.sampling_rate/1000))
        self.good_matches = matches['index1']

        garbage_matches = ~np.in1d(np.arange(len(times2)), self.good_matches)
        garbage_channels = self.peaks['channel_ind'][garbage_matches]
        garbage_peaks = times2[garbage_matches]
        nb_garbage = len(garbage_peaks)

        ratio = 100*len(garbage_peaks)/len(times2)
        self.garbage_sorting = NumpySorting.from_times_labels(garbage_peaks, garbage_channels, self.sampling_rate)
        
        print("The peaks have {0:.2f}% of garbage (without gt around)".format(ratio))

        self.comp = GroundTruthComparison(self.gt_sorting, self.sliced_gt_sorting, exhaustive_gt=self.exhaustive_gt)

        for label, sorting in zip(['gt', 'full_gt', 'garbage'], [self.sliced_gt_sorting, self.gt_sorting, self.garbage_sorting]): 

            tmp_folder = os.path.join(self.tmp_folder, label)
            if os.path.exists(tmp_folder):
                import shutil
                shutil.rmtree(tmp_folder)

            if not (label == 'full_gt' and label in self.waveforms):

                if self.verbose:
                    print(f"Extracting waveforms for {label}")

                self.waveforms[label] = extract_waveforms(self.recording_f, sorting, tmp_folder, load_if_exists=True,
                                       ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500, return_scaled=False, 
                                       **self.job_kwargs)

                self.templates[label] = self.waveforms[label].get_all_templates(mode='median')
    
        if self.gt_peaks is None:
            if self.verbose:
                print("Computing gt peaks")
            gt_peaks_ = self.gt_sorting.to_spike_vector()
            self.gt_peaks = np.zeros(gt_peaks_.size, dtype=[('sample_ind', '<i8'), ('channel_ind', '<i8'), ('segment_ind', '<i8'), ('amplitude', '<f8')])
            self.gt_peaks['sample_ind'] = gt_peaks_['sample_ind']
            self.gt_peaks['segment_ind'] = gt_peaks_['segment_ind']
            max_channels = get_template_extremum_channel(self.waveforms['full_gt'], peak_sign='neg', outputs='index')
            max_amplitudes = get_template_extremum_amplitude(self.waveforms['full_gt'], peak_sign='neg')

            for unit_ind, unit_id in enumerate(self.waveforms['full_gt'].sorting.unit_ids):
                mask = gt_peaks_['unit_ind'] == unit_ind
                max_channel = max_channels[unit_id]
                self.gt_peaks['channel_ind'][mask] = max_channel
                self.gt_peaks['amplitude'][mask] = max_amplitudes[unit_id]

        self.sliced_gt_peaks = self.gt_peaks[gt_matches]
        self.sliced_gt_positions = self.gt_positions[gt_matches]
        self.sliced_gt_labels = self.sliced_gt_sorting.to_spike_vector()['unit_ind']
        self.gt_labels = self.gt_sorting.to_spike_vector()['unit_ind']
        self.garbage_positions = self.positions[garbage_matches]
        self.garbage_peaks = self.peaks[garbage_matches]


    