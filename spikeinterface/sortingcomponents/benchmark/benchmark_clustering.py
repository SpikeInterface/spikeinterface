
from spikeinterface.core import extract_waveforms
from spikeinterface.toolkit import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.extractors import read_mearec
from spikeinterface.core import NumpySorting
from spikeinterface.toolkit.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import plot_probe_map, plot_agreement_matrix, plot_comparison_collision_by_similarity, plot_unit_templates, plot_unit_waveforms
from spikeinterface.toolkit.postprocessing import compute_principal_components
from spikeinterface.comparison.comparisontools import make_matching_events

import time
import string, random
import pylab as plt
import os

class BenchmarkClustering:

    def __init__(self, mearec_file, method, tmp_folder=None, job_kwargs={}, verbose=True):
        self.mearec_file = mearec_file
        self.method = method
        self.verbose = verbose
        self.recording, self.gt_sorting = read_mearec(mearec_file)
        self.recording_f = bandpass_filter(self.recording, dtype='float32')
        self.recording_f = common_reference(self.recording_f)
        self.sampling_rate = self.recording_f.get_sampling_frequency()
        self.job_kwargs = job_kwargs

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', 'clustering')

        self._peaks = None
        self._selected_peaks = None
        self._positions = None

    @property
    def peaks(self):
        if self._peaks is None:
            self.detect_peaks()
        return self._peaks

    @property
    def selected_peaks(self):
        if self._selected_peaks is None:
            self.select_peaks()
        return self._selected_peaks

    @property
    def positions(self):
        if self._positions is None:
            self.localize_peaks()
        return self._positions

    def detect_peaks(self, method_kwargs={'method' : 'locally_exclusive'}):
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Detecting peaks with method {method}')
        self._peaks = detect_peaks(self.recording_f, **method_kwargs, **self.job_kwargs)

    def select_peaks(self, method_kwargs = {'method' : 'uniform', 'n_peaks' : 100}):
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Selecting peaks with method {method}')
        self._selected_peaks = select_peaks(self.peaks, **method_kwargs, **self.job_kwargs)
        if self.verbose:
            ratio = len(self._selected_peaks)/len(self.peaks)
            print(f'The ratio of peaks kept for clustering is {ratio}%')

    def localize_peaks(self, method_kwargs = {'method' : 'center_of_mass'}):
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Localizing peaks with method {method}')
        self._positions = localize_peaks(self.recording_f, self.selected_peaks, **method_kwargs, **self.job_kwargs)

    def run(self, peaks=None, positions=None, method_kwargs={}):
        t_start = time.time()
        if self.verbose:
            print(f'Launching the {self.method} clustering algorithm')
        if peaks is not None:
            self._peaks = peaks
            self._selected_peaks = peaks
        if positions is not None:
            self._positions = positions
        labels, peak_labels = find_cluster_from_peaks(self.recording_f, self.selected_peaks, method=self.method, method_kwargs=method_kwargs, **self.job_kwargs)
        self.noise = peak_labels == -1
        self.run_time = time.time() - t_start
        self.selected_peaks_labels = peak_labels
        self.labels = labels

        self.clustering = NumpySorting.from_times_labels(self.selected_peaks['sample_ind'][~self.noise], self.selected_peaks_labels[~self.noise], self.sampling_rate)
        if self.verbose:
            print("Performing the comparison with (sliced) ground truth")

        times1 = self.gt_sorting.get_all_spike_trains()[0]
        times2 = self.clustering.get_all_spike_trains()[0]
        matches = make_matching_events(times1[0], times2[0], int(0.1*self.sampling_rate/1000))

        idx = matches['index1']
        self.sliced_gt_sorting = NumpySorting.from_times_labels(times1[0][idx], times1[1][idx], self.sampling_rate)

        self.comp = GroundTruthComparison(self.sliced_gt_sorting, self.clustering)

        self.waveforms = {}
        self.pcas = {}
        self.templates = {}
        if self.verbose:
            print("Extracting waveforms")

        for label, sorting in zip(['gt', 'clustering'], [self.sliced_gt_sorting, self.clustering]):
            tmp_folder = os.path.join(self.tmp_folder, label)
            if os.path.exists(tmp_folder):
                import shutil
                shutil.rmtree(tmp_folder)
            self.waveforms[label] = extract_waveforms(self.recording_f, sorting, tmp_folder, load_if_exists=True,
                                       ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500,
                                       **self.job_kwargs)


            #self.pcas[label] = compute_principal_components(self.waveforms[label], load_if_exists=True,
            #                     n_components=5, mode='by_channel_local',
            #                     whiten=True, dtype='float32')

            self.templates[label] = self.waveforms[label].get_all_templates(mode='median')
    

    def plot(self, title=None, metric='cosine', show_probe=True):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)
        ax.scatter(self.positions['x'][~self.noise], self.positions['y'][~self.noise], c=self.selected_peaks_labels[~self.noise], s=1, alpha=0.5)
        ax.scatter(self.positions['x'][self.noise], self.positions['y'][self.noise], c='k', s=1, alpha=0.1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        metrics = compute_quality_metrics(self.waveforms['gt'], metric_names=['snr'], load_if_exists=False)

        ax = axs[0, 1]
        plot_agreement_matrix(self.comp, ax=ax)

        scores = self.comp.get_ordered_agreement_scores()
        unit_ids1 = scores.index.values
        ids_2 = scores.columns.values
        ids_1 = self.comp.sorting1.ids_to_indices(unit_ids1)


        a = self.templates['gt'].reshape(len(self.templates['gt']), -1)[ids_1]
        b = self.templates['clustering'].reshape(len(self.templates['clustering']), -1)[ids_2]
        

        import sklearn
        distances = sklearn.metrics.pairwise_distances(a, b, metric)
        ax = axs[1, 0]
        im = ax.imshow(distances, aspect='auto')
        ax.set_title(metric)
        fig.colorbar(im, ax=ax)

        ax = axs[1, 1]
        for performance_name in ['accuracy', 'recall', 'precision']:
            perf = self.comp.get_performance()[performance_name]
            ax.plot(metrics['snr'], perf, markersize=10, marker='.', ls='', label=performance_name)
        ax.set_xlabel('snr')
        ax.set_ylabel('performance')
        ax.legend()