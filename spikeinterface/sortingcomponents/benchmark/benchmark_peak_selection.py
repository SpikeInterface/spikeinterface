
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
from spikeinterface.toolkit.postprocessing import get_template_extremum_channel, get_template_extremum_amplitude

import time
import string, random
import pylab as plt
import os
import numpy as np

class BenchmarkPeakSelection:

    def __init__(self, mearec_file, job_kwargs={}, tmp_folder=None, verbose=True):
        self.mearec_file = mearec_file
        self.verbose = verbose
        self.recording, self.gt_sorting = read_mearec(mearec_file)
        self.job_kwargs = job_kwargs
        self.recording_f = bandpass_filter(self.recording, dtype='float32')
        self.recording_f = common_reference(self.recording_f)
        self.sampling_rate = self.recording_f.get_sampling_frequency()

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self._peaks = None
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


    @property
    def peaks(self):
        if self._peaks is None:
            self.detect_peaks()
        return self._peaks

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

        #print(len(times1[0]), len(matches['index1']))
        gt_matches = matches['index1']
        sorting_key = lambda x: int(''.join(filter(str.isdigit, x)))
        self.sliced_gt_sorting = NumpySorting.from_times_labels(times1[0][gt_matches], times1[1][gt_matches], self.sampling_rate, sorting_key=sorting_key)
        ratio = 100*len(gt_matches)/len(times1[0])
        print("Only {0:.2f}% of gt peaks are matched to detected peaks".format(ratio))

        matches = make_matching_events(times2, times1[0], int(delta*self.sampling_rate/1000))
        good_matches = matches['index1']
        garbage_matches = ~np.in1d(np.arange(len(times2)), good_matches)
        garbage_channels = self.peaks['channel_ind'][garbage_matches]
        garbage_peaks = times2[garbage_matches]
        nb_garbage = len(garbage_peaks)

        ratio = 100*len(garbage_peaks)/len(times2)
        self.garbage_sorting = NumpySorting.from_times_labels(garbage_peaks, garbage_channels, self.sampling_rate)
        
        print("The peaks have {0:.2f}% of garbage (without gt around)".format(ratio))

        self.comp = GroundTruthComparison(self.gt_sorting, self.sliced_gt_sorting)

        for label, sorting in zip(['gt', 'full_gt', 'garbage'], [self.sliced_gt_sorting, self.gt_sorting, self.garbage_sorting]): 

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


    def _get_colors(self, sorting, excluded_ids=[-1]):
        from spikeinterface.widgets import get_unit_colors
        colors = get_unit_colors(sorting)
        result = {}
        for key, value in colors.items():
            result[sorting.id_to_index(key)] = value
        for key in excluded_ids:
            result[key] = 'k'
        return result

    def _get_labels(self, sorting, excluded_ids={-1}):
        result = {}
        for unid_id in sorting.unit_ids:
            result[sorting.id_to_index(unid_id)] = unid_id
        for key in excluded_ids:
            result[key] = 'noise'
        return result

    def _scatter_clusters(self, xs, ys, sorting, colors=None, labels=None, ax=None, n_std=2.0, excluded_ids=[-1], s=1, alpha=0.5):

        if colors is None:
            colors = self._get_colors(sorting, excluded_ids)
        if labels is None:
            labels = self._get_labels(sorting, excluded_ids)

        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        ax = ax or plt.gca()
        # scatter and collect gaussian info
        means = {}
        covs = {}
        labels_ids = sorting.get_all_spike_trains()[0][1]
        ids = sorting.ids_to_indices(labels_ids)

        for k in np.unique(ids):
            where = np.flatnonzero(ids == k)
            xk = xs[where]
            yk = ys[where]
            ax.scatter(xk, yk, s=s, color=colors[k], alpha=alpha, marker=".")
            if k not in excluded_ids:
                x_mean, y_mean = xk.mean(), yk.mean()
                xycov = np.cov(xk, yk)
                means[k] = x_mean, y_mean
                covs[k] = xycov
                ax.annotate(labels[k], (x_mean, y_mean))

        for k in means.keys():
            mean_x, mean_y = means[k]
            cov = covs[k]

            with np.errstate(invalid="ignore"):
                vx, vy = cov[0, 0], cov[1, 1]
                rho = cov[0, 1] / np.sqrt(vx * vy)
            if not np.isfinite([vx, vy, rho]).all():
                continue

            ell = Ellipse(
                (0, 0),
                width=2 * np.sqrt(1 + rho),
                height=2 * np.sqrt(1 - rho),
                facecolor=(0, 0, 0, 0),
                edgecolor=colors[k],
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

    def plot_clusters(self, title=None, show_probe=False):
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
        if title is not None:
            fig.suptitle(f'Peak selection results with {title}')

        ax = axs[0]
        ax.set_title('Full gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)

        colors = self._get_colors(self.gt_sorting)
        self._scatter_clusters(self.gt_positions['x'], self.gt_positions['y'],  self.gt_sorting, colors, s=1, alpha=0.5, ax=ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axs[1]
        ax.set_title('Sliced gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)

        self._scatter_clusters(self.sliced_gt_positions['x'], self.sliced_gt_positions['y'],  self.sliced_gt_sorting, colors, s=1, alpha=0.5, ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x')
        ax.set_yticks([], [])

        ax = axs[2]
        ax.set_title('Garbage')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)

        ax.scatter(self.garbage_positions['x'], self.garbage_positions['y'],  c='k', s=1, alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x')
        ax.set_yticks([], [])


    def plot_clusters_amplitudes(self, title=None, show_probe=False, clim=(-100, 0), cmap='viridis'):
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
        if title is not None:
            fig.suptitle(f'Peak selection results with {title}')

        ax = axs[0]
        ax.set_title('Full gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)
        
        from spikeinterface.widgets import get_unit_colors
        channels = get_template_extremum_channel(self.waveforms['full_gt'], outputs='index')

        #cb = fig.colorbar(cm, ax=ax)
        #cb.set_label(metric)

        import matplotlib
        my_cmap = plt.get_cmap(cmap)
        cNorm  = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)



        for unit_id in self.gt_sorting.unit_ids:
            wfs = self.waveforms['full_gt'].get_waveforms(unit_id)
            amplitudes = wfs[:, self.waveforms['full_gt'].nbefore, channels[unit_id]]

            idx = self.waveforms['full_gt'].get_sampled_indices(unit_id)['spike_index']
            all_spikes = self.waveforms['full_gt'].sorting.get_unit_spike_train(unit_id)
            mask = np.in1d(self.gt_peaks['sample_ind'], all_spikes[idx])
            colors = scalarMap.to_rgba(self.gt_peaks['amplitude'][mask])
            ax.scatter(self.gt_positions['x'][mask], self.gt_positions['y'][mask], c=colors, s=1, alpha=0.5)
            x_mean, y_mean = (self.gt_positions['x'][mask].mean(), self.gt_positions['y'][mask].mean())
            ax.annotate(unit_id, (x_mean, y_mean))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axs[1]
        ax.set_title('Sliced gt clusters')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)
        
        from spikeinterface.widgets import get_unit_colors
        channels = get_template_extremum_channel(self.waveforms['gt'], outputs='index')

        for unit_id in self.sliced_gt_sorting.unit_ids:
            wfs = self.waveforms['gt'].get_waveforms(unit_id)
            amplitudes = wfs[:, self.waveforms['gt'].nbefore, channels[unit_id]]

            idx = self.waveforms['gt'].get_sampled_indices(unit_id)['spike_index']
            all_spikes = self.waveforms['gt'].sorting.get_unit_spike_train(unit_id)
            mask = np.in1d(self.sliced_gt_peaks['sample_ind'], all_spikes[idx])
            colors = scalarMap.to_rgba(self.sliced_gt_peaks['amplitude'][mask])
            ax.scatter(self.sliced_gt_positions['x'][mask], self.sliced_gt_positions['y'][mask],  c=colors, s=1, alpha=0.5)
            x_mean, y_mean = (self.sliced_gt_positions['x'][mask].mean(), self.sliced_gt_positions['y'][mask].mean())
            ax.annotate(unit_id, (x_mean, y_mean))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('x')
        ax.set_yticks([], [])
        #ax.set_ylabel('y')

        ax = axs[2]
        ax.set_title('Garbage')
        if show_probe:
            plot_probe_map(self.recording_f, ax=ax)
        
        from spikeinterface.widgets import get_unit_colors
        channels = get_template_extremum_channel(self.waveforms['garbage'], outputs='index')

        for unit_id in self.garbage_sorting.unit_ids:
            wfs = self.waveforms['garbage'].get_waveforms(unit_id)
            amplitudes = wfs[:, self.waveforms['garbage'].nbefore, channels[unit_id]]

            idx = self.waveforms['garbage'].get_sampled_indices(unit_id)['spike_index']
            all_spikes = self.waveforms['garbage'].sorting.get_unit_spike_train(unit_id)
            mask = np.in1d(self.garbage_peaks['sample_ind'], all_spikes[idx])
            colors = scalarMap.to_rgba(self.garbage_peaks['amplitude'][mask])
            ax.scatter(self.garbage_positions['x'][mask], self.garbage_positions['y'][mask],  c=colors, s=1, alpha=0.5)
            x_mean, y_mean = (self.garbage_positions['x'][mask].mean(), self.garbage_positions['y'][mask].mean())
            ax.annotate(unit_id, (x_mean, y_mean))



        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlabel('x')
        ax.set_yticks([], [])
        #ax.set_ylabel('y')