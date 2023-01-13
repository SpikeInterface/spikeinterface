
from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.widgets import plot_probe_map, plot_agreement_matrix, plot_comparison_collision_by_similarity, plot_unit_templates, plot_unit_waveforms
from spikeinterface.postprocessing import compute_spike_locations
from spikeinterface.postprocessing.unit_localization import compute_center_of_mass, compute_monopolar_triangulation
from spikeinterface.core import get_noise_levels

import time
import string, random
import pylab as plt
import os
import numpy as np

class BenchmarkPeakLocalization:

    def __init__(self, recording, gt_sorting, gt_positions, job_kwargs={}, tmp_folder=None, verbose=True, title=None):
        self.verbose = verbose
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.sampling_rate = self.recording.get_sampling_frequency()
        self.title = title

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self.gt_positions = gt_positions
        self.waveforms = extract_waveforms(self.recording, self.gt_sorting, self.tmp_folder,
                                   ms_before=2.5, ms_after=2.5, max_spikes_per_unit=500, return_scaled=False,
                                   **self.job_kwargs)

    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_folder)

    def run(self, method, method_kwargs={}):
        t_start = time.time()
        if self.title is None:
            self.title = method
        if method == 'center_of_mass':
            self.template_positions = compute_center_of_mass(self.waveforms, **method_kwargs)
        elif method == 'monopolar_triangulation':
            self.template_positions = compute_monopolar_triangulation(self.waveforms, **method_kwargs)[:,:2]
        self.spike_positions = compute_spike_locations(self.waveforms, method, method_kwargs=method_kwargs, **self.job_kwargs)

        self.raw_templates_results = {}
        all_times = self.waveforms.sorting.get_all_spike_trains()[0][0]

        for unit_ind, unit_id in enumerate(self.waveforms.sorting.unit_ids):
            times = self.waveforms.sorting.get_unit_spike_train(unit_id)
            mask = np.in1d(all_times, times)
            data = self.spike_positions[mask]
            self.raw_templates_results[unit_id] = np.sqrt((data['x'] - self.gt_positions[unit_ind, 0])**2 + (data['y'] - self.gt_positions[unit_ind, 1])**2)

        self.means_over_templates = np.array([np.mean(self.raw_templates_results[unit_id]) for unit_id in  self.waveforms.sorting.unit_ids])
        self.stds_over_templates = np.array([np.std(self.raw_templates_results[unit_id]) for unit_id in  self.waveforms.sorting.unit_ids])

    def plot_template_errors(self, show_probe=True):
        import spikeinterface.full as si
        import pylab as plt
        si.plot_probe_map(self.recording)
        plt.scatter(self.gt_positions[:, 0], self.gt_positions[:, 1], c=np.arange(len(self.gt_positions)), cmap='jet')
        plt.scatter(self.template_positions[:, 0], self.template_positions[:, 1], c=np.arange(len(self.template_positions)), cmap='jet', marker='v')


def plot_comparison_positions(benchmarks, mode='average'):
    norms = np.linalg.norm(benchmarks[0].waveforms.get_all_templates(mode=mode),  axis=(1, 2))
    idx = np.argsort(norms)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    ax = axs[0, 0]
    #ax.set_title(title)

    for bench in benchmarks:
        errors = np.linalg.norm(bench.template_positions - bench.gt_positions, axis=1)
        ax.plot(norms, errors, '.', label=bench.title)

    ax.legend()
    ax.set_xlabel('norms')
    ax.set_ylabel('error')

    ax = axs[0, 1]

    for bench in benchmarks:
        errors = np.linalg.norm(bench.template_positions - bench.gt_positions, axis=1)
        ax.plot(np.linalg.norm(bench.gt_positions, axis=1), errors, '.', label=bench.title)

    ax.set_xlabel('distance to center')
    ax.set_ylabel('error')

    ax = axs[1, 0]

    for bench in benchmarks:
        ax.plot(bench.means_over_templates[idx], lw=2, label=bench.title)
        ymin = (bench.means_over_templates - bench.stds_over_templates)[idx]
        ymax = (bench.means_over_templates + bench.stds_over_templates)[idx]

        ax.fill_between(np.arange(len(idx)), ymin, ymax, alpha=0.5)

    ax.set_xlabel('distance to center')
    ax.set_ylabel('error')
