
from spikeinterface.core import extract_waveforms
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks, clustering_methods
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.extractors import read_mearec
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import plot_probe_map, plot_agreement_matrix, plot_comparison_collision_by_similarity, plot_unit_templates, plot_unit_waveforms
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.comparison.comparisontools import make_matching_events
from spikeinterface.postprocessing import get_template_extremum_channel
from spikeinterface.core import get_noise_levels 

import time
import string, random
import pylab as plt
import os
import numpy as np

class BenchmarkClustering:

    def __init__(self, recording, gt_sorting, method, exhaustive_gt=True, tmp_folder=None, job_kwargs={}, verbose=True):

        self.method = method

        assert method in clustering_methods, "Clustering method should be in %s" %clustering_methods.keys()

        self.verbose = verbose
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.exhaustive_gt = exhaustive_gt
        self.recording_f = bandpass_filter(self.recording,  dtype='float32')
        self.recording_f = common_reference(self.recording_f)
        self.sampling_rate = self.recording_f.get_sampling_frequency()
        self.job_kwargs = job_kwargs

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self._peaks = None
        self._selected_peaks = None
        self._positions = None
        self._gt_positions = None
        self.gt_peaks = None

        self.waveforms = {}
        self.pcas = {}
        self.templates = {}

    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_folder)

    def set_peaks(self, peaks):
        self._peaks = peaks

    def set_positions(self, positions):
        self._positions = positions

    def set_gt_positions(self, gt_positions):
        self._gt_positions = gt_positions

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

    def localize_gt_peaks(self, method_kwargs = {'method' : 'center_of_mass'}):
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        if self.verbose:
            method = method_kwargs['method']
            print(f'Localizing gt peaks with method {method}')
        self._gt_positions = localize_peaks(self.recording_f, self.gt_peaks, **method_kwargs, **self.job_kwargs)

    def run(self, peaks=None, positions=None, method=None, method_kwargs={}, delta=0.2):
        t_start = time.time()
        if method is not None:
            self.method = method
        if peaks is not None:
            self._peaks = peaks
            self._selected_peaks = peaks

        nb_peaks = len(self.selected_peaks)
        if self.verbose:
            print(f'Launching the {self.method} clustering algorithm with {nb_peaks} peaks')

        if positions is not None:
            self._positions = positions

        labels, peak_labels = find_cluster_from_peaks(self.recording_f, self.selected_peaks, method=self.method, method_kwargs=method_kwargs, **self.job_kwargs)
        nb_clusters = len(labels)
        if self.verbose:
            print(f'{nb_clusters} clusters have been found')
        self.noise = peak_labels == -1
        self.run_time = time.time() - t_start
        self.selected_peaks_labels = peak_labels
        self.labels = labels

        
        self.clustering = NumpySorting.from_times_labels(self.selected_peaks['sample_ind'][~self.noise], self.selected_peaks_labels[~self.noise], self.sampling_rate)
        if self.verbose:
            print("Performing the comparison with (sliced) ground truth")

        times1 = self.gt_sorting.get_all_spike_trains()[0]
        times2 = self.clustering.get_all_spike_trains()[0]
        matches = make_matching_events(times1[0], times2[0], int(delta*self.sampling_rate/1000))

        self.matches = matches
        idx = matches['index1']
        self.sliced_gt_sorting = NumpySorting.from_times_labels(times1[0][idx], times1[1][idx], self.sampling_rate, unit_ids = self.gt_sorting.unit_ids)

        self.comp = GroundTruthComparison(self.sliced_gt_sorting, self.clustering, exhaustive_gt=self.exhaustive_gt)

        for label, sorting in zip(['gt', 'clustering', 'full_gt'], [self.sliced_gt_sorting, self.clustering, self.gt_sorting]): 

            tmp_folder = os.path.join(self.tmp_folder, label)
            if os.path.exists(tmp_folder):
                import shutil
                shutil.rmtree(tmp_folder)

            if not (label == 'full_gt' and label in self.waveforms):

                if self.verbose:
                    print(f"Extracting waveforms for {label}")

                self.waveforms[label] = extract_waveforms(self.recording_f, sorting, tmp_folder, load_if_exists=True,
                                       ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500,
                                       **self.job_kwargs)

                #self.pcas[label] = compute_principal_components(self.waveforms[label], load_if_exists=True,
                #                     n_components=5, mode='by_channel_local',
                #                     whiten=True, dtype='float32')

                self.templates[label] = self.waveforms[label].get_all_templates(mode='median')
    
        if self.gt_peaks is None:
            if self.verbose:
                print("Computing gt peaks")
            gt_peaks_ = self.gt_sorting.to_spike_vector()
            self.gt_peaks = np.zeros(gt_peaks_.size, dtype=[('sample_ind', '<i8'), ('channel_ind', '<i8'), ('segment_ind', '<i8')])
            self.gt_peaks['sample_ind'] = gt_peaks_['sample_ind']
            self.gt_peaks['segment_ind'] = gt_peaks_['segment_ind']
            max_channels = get_template_extremum_channel(self.waveforms['full_gt'], peak_sign='neg', outputs='index')

            for unit_ind, unit_id in enumerate(self.waveforms['full_gt'].sorting.unit_ids):
                mask = gt_peaks_['unit_ind'] == unit_ind
                max_channel = max_channels[unit_id]
                self.gt_peaks['channel_ind'][mask] = max_channel

        self.sliced_gt_peaks = self.gt_peaks[idx]
        self.sliced_gt_positions = self.gt_positions[idx]
        self.sliced_gt_labels = self.sliced_gt_sorting.to_spike_vector()['unit_ind']
        self.gt_labels = self.gt_sorting.to_spike_vector()['unit_ind']


    def _scatter_clusters(self, xs, ys, sorting, colors=None, labels=None, ax=None, n_std=2.0, force_black_for=[], s=1, alpha=0.5, show_ellipses=True):

        if colors is None:
            from spikeinterface.widgets import get_unit_colors
            colors = get_unit_colors(sorting)

        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        ax = ax or plt.gca()
        # scatter and collect gaussian info
        means = {}
        covs = {}
        labels_ids = sorting.get_all_spike_trains()[0][1]

        for unit_ind, unit_id in enumerate(sorting.unit_ids):
            where = np.flatnonzero(labels_ids == unit_id)
            xk = xs[where]
            yk = ys[where]
            
            if unit_id not in force_black_for:
                ax.scatter(xk, yk, s=s, color=colors[unit_id], alpha=alpha, marker=".")
                x_mean, y_mean = xk.mean(), yk.mean()
                xycov = np.cov(xk, yk)
                means[unit_id] = x_mean, y_mean
                covs[unit_id] = xycov
                ax.annotate(unit_id, (x_mean, y_mean))
            else:
                ax.scatter(xk, yk, s=s, color='k', alpha=alpha, marker=".")

        for unit_id in means.keys():
            mean_x, mean_y = means[unit_id]
            cov = covs[unit_id]

            with np.errstate(invalid="ignore"):
                vx, vy = cov[0, 0], cov[1, 1]
                rho = cov[0, 1] / np.sqrt(vx * vy)
            if not np.isfinite([vx, vy, rho]).all():
                continue

            if show_ellipses:
                ell = Ellipse(
                    (0, 0),
                    width=2 * np.sqrt(1 + rho),
                    height=2 * np.sqrt(1 - rho),
                    facecolor=(0, 0, 0, 0),
                    edgecolor=colors[unit_id],
                    linewidth=1,
                )
                transform = (
                    transforms.Affine2D()
                    .rotate_deg(45)
                    .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
                    .translate(mean_x, mean_y)
                )
                ell.set_transform(transform + ax.transData)
                ax.add_patch(ell)


    def plot_clusters(self, show_probe=True, show_ellipses=True):

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
        fig.suptitle(f'Clustering results with {self.method}')
        ax = axs[0]
        ax.set_title('Full gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)

        from spikeinterface.widgets import get_unit_colors
        colors = get_unit_colors(self.gt_sorting)
        self._scatter_clusters(self.gt_positions['x'], self.gt_positions['y'], self.gt_sorting, colors, s=1, alpha=0.5, ax=ax, show_ellipses=show_ellipses)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axs[1]
        ax.set_title('Sliced gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)

        self._scatter_clusters(self.sliced_gt_positions['x'], self.sliced_gt_positions['y'], self.sliced_gt_sorting, colors, s=1, alpha=0.5, ax=ax, show_ellipses=show_ellipses)
        if self.exhaustive_gt:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_xlabel('x')
        ax.set_yticks([], [])

        ax = axs[2]
        ax.set_title('Found clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)
        ax.scatter(self.positions['x'][self.noise], self.positions['y'][self.noise], c='k', s=1, alpha=0.1)
        self._scatter_clusters(self.positions['x'][~self.noise], self.positions['y'][~self.noise], self.clustering, s=1, alpha=0.5, ax=ax, show_ellipses=show_ellipses)
        
        ax.set_xlabel('x')
        if self.exhaustive_gt:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_yticks([], [])


    def plot_statistics(self, metric='cosine', annotations=True, detect_threshold=5):

        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
        
        fig.suptitle(f'Clustering results with {self.method}')
        metrics = compute_quality_metrics(self.waveforms['gt'], metric_names=['snr'], load_if_exists=False)

        ax = axs[0, 0]
        plot_agreement_matrix(self.comp, ax=ax)
        scores = self.comp.get_ordered_agreement_scores()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        unit_ids1 = scores.index.values
        unit_ids2 = scores.columns.values
        inds_1 = self.comp.sorting1.ids_to_indices(unit_ids1)
        snrs = metrics['snr'][inds_1]

        nb_detectable = len(unit_ids1)

        if detect_threshold is not None:
            for count, snr in enumerate(snrs):
                if snr < detect_threshold:
                    ax.plot([xmin, xmax], [count, count], 'k')
                    nb_detectable -= 1

        ax.plot([nb_detectable+0.5, nb_detectable+0.5], [ymin, ymax], 'r')

        # import MEArec as mr
        # mearec_recording = mr.load_recordings(self.mearec_file)
        # positions = mearec_recording.template_locations[:]

        # self.found_positions = np.zeros((len(self.labels), 2))
        # for i in range(len(self.labels)):
        #     data = self.positions[self.selected_peaks_labels == self.labels[i]]
        #     self.found_positions[i] = np.median(data['x']), np.median(data['y'])

        
        unit_ids1 = scores.index.values
        unit_ids2 = scores.columns.values
        inds_1 = self.comp.sorting1.ids_to_indices(unit_ids1)
        inds_2 = self.comp.sorting2.ids_to_indices(unit_ids2)

        a = self.templates['gt'].reshape(len(self.templates['gt']), -1)[inds_1]
        b = self.templates['clustering'].reshape(len(self.templates['clustering']), -1)[inds_2]
        
        import sklearn
        if metric == 'cosine':
            distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
        else:
            distances = sklearn.metrics.pairwise_distances(a, b, metric)

        ax = axs[0, 1]
        nb_peaks = np.array([len(self.sliced_gt_sorting.get_unit_spike_train(i)) for i in self.sliced_gt_sorting.unit_ids])

        nb_potentials = np.sum(scores.max(1).values > 0.1)

        ax.plot(metrics['snr'][unit_ids1][inds_1[:nb_potentials]], nb_peaks[inds_1[:nb_potentials]], markersize=10, marker='.', ls='', c='k', label='Cluster potentially found')
        ax.plot(metrics['snr'][unit_ids1][inds_1[nb_potentials:]], nb_peaks[inds_1[nb_potentials:]], markersize=10, marker='.', ls='', c='r', label='Cluster clearly missed')

        if annotations:
            for l,x,y in zip(unit_ids1[:len(inds_2)], metrics['snr'][unit_ids1][inds_1[:len(inds_2)]], nb_peaks[inds_1[:len(inds_2)]]):
                ax.annotate(l, (x, y))

            for l,x,y in zip(unit_ids1[len(inds_2):], metrics['snr'][unit_ids1][inds_1[len(inds_2):]], nb_peaks[inds_1[len(inds_2):]]):
                ax.annotate(l, (x, y),c='r')

        if detect_threshold is not None:
            ymin, ymax = ax.get_ylim()
            ax.plot([detect_threshold, detect_threshold], [ymin, ymax], 'k--')

        ax.legend()
        ax.set_xlabel('template snr')
        ax.set_ylabel('nb spikes')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = axs[0, 2]
        im = ax.imshow(distances, aspect='auto')
        ax.set_title(metric)
        fig.colorbar(im, ax=ax)

        if detect_threshold is not None:
            for count, snr in enumerate(snrs):
                if snr < detect_threshold:
                    ax.plot([xmin, xmax], [count, count], 'w')

            ymin, ymax = ax.get_ylim()
            ax.plot([nb_detectable+0.5, nb_detectable+0.5], [ymin, ymax], 'r')


        ax.set_yticks(np.arange(0, len(scores.index)))
        ax.set_yticklabels(scores.index, fontsize=8)

        res = []
        nb_spikes = []
        energy = []
        nb_channels = []

        
        noise_levels = get_noise_levels(self.recording_f, return_scaled=False)

        for found, real in zip(unit_ids2, unit_ids1):
            wfs = self.waveforms['clustering'].get_waveforms(found)
            wfs_real = self.waveforms['gt'].get_waveforms(real)
            template = self.waveforms['clustering'].get_template(found)
            template_real = self.waveforms['gt'].get_template(real)
            nb_channels += [np.sum(np.std(template_real, 0) < noise_levels)]

            wfs = wfs.reshape(len(wfs), -1)
            template = template.reshape(template.size, 1).T
            template_real = template_real.reshape(template_real.size, 1).T

            if metric == 'cosine':
                dist = sklearn.metrics.pairwise.cosine_similarity(template, template_real, metric).flatten().tolist()
            else:
                dist = sklearn.metrics.pairwise_distances(template, template_real, metric).flatten().tolist()
            res += dist
            nb_spikes += [self.sliced_gt_sorting.get_unit_spike_train(real).size]
            energy += [np.linalg.norm(template_real)]


        ax = axs[1, 0]
        res = np.array(res)
        nb_spikes = np.array(nb_spikes)
        nb_channels = np.array(nb_channels)
        energy = np.array(energy)

        snrs = metrics['snr'][unit_ids1][inds_1[:len(inds_2)]]
        cm = ax.scatter(snrs, nb_spikes, c=res)
        ax.set_xlabel('template snr')
        ax.set_ylabel('nb spikes')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cb = fig.colorbar(cm, ax=ax)
        cb.set_label(metric)
        if detect_threshold is not None:
            ymin, ymax = ax.get_ylim()
            ax.plot([detect_threshold, detect_threshold], [ymin, ymax], 'k--')

        if annotations:
            for l,x,y in zip(unit_ids1[:len(inds_2)], snrs, nb_spikes):
                ax.annotate(l, (x, y))

        ax = axs[1, 1]
        cm = ax.scatter(energy, nb_channels, c=res)
        ax.set_xlabel('template energy')
        ax.set_ylabel('nb channels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cb = fig.colorbar(cm, ax=ax)
        cb.set_label(metric)
        
        if annotations:
            for l,x,y in zip(unit_ids1[:len(inds_2)], energy, nb_channels):
                ax.annotate(l, (x, y))


        ax = axs[1, 2]
        for performance_name in ['accuracy', 'recall', 'precision']:
            perf = self.comp.get_performance()[performance_name]
            ax.plot(metrics['snr'], perf, markersize=10, marker='.', ls='', label=performance_name)
        ax.set_xlabel('template snr')
        ax.set_ylabel('performance')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        if detect_threshold is not None:
            ymin, ymax = ax.get_ylim()
            ax.plot([detect_threshold, detect_threshold], [ymin, ymax], 'k--')

        plt.tight_layout()