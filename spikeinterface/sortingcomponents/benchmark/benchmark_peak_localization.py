
from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.widgets import plot_probe_map, plot_agreement_matrix, plot_comparison_collision_by_similarity, plot_unit_templates, plot_unit_waveforms
from spikeinterface.postprocessing import get_template_extremum_channel, get_template_extremum_amplitude, compute_center_of_mass
from spikeinterface.core import get_noise_levels

import time
import string, random
import pylab as plt
import os
import numpy as np

class BenchmarkPeakLocalization:

    def __init__(self, recording, gt_sorting, gt_positions, job_kwargs={}, tmp_folder=None, verbose=True):
        self.verbose = verbose
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.sampling_rate = self.recording.get_sampling_frequency()

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self.gt_positions = gt_positions
        self.waveforms = extract_waveforms(self.recording, self.gt_sorting, self.tmp_folder, load_if_exists=False,
                                   ms_before=2.5, ms_after=2.5, max_spikes_per_unit=500, return_scaled=False,
                                   **self.job_kwargs)

    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_folder)

    def localize_peaks(self, method_kwargs = {'method' : 'center_of_mass'}):
        import spikeinterface.full as si
        self.spike_positions = si.compute_spike_locations(self.waveforms, **method_kwargs)

    def compute_template_center_of_mass(self, method_kwargs = {}):
        self.template_positions = compute_center_of_mass(self.waveforms, **method_kwargs)

    def compute_template_monopolar(self, method_kwargs = {}):
        self.template_positions = compute_monopolar_triangulation(self.waveforms, **method_kwargs)

    def run(self, num_spikes, localize_peaks):
        t_start = time.time()
        self._positions = localize_peaks(self.recording_f, self.peaks, **method_kwargs, **self.job_kwargs)

    def plot_template_errors(self, show_probe=True):
        import spikeinterface.full as si
        import pylab as plt
        si.plot_probe_map(self.recording)
        plt.scatter(self.gt_positions[:, 0], self.gt_positions[:, 1], c=np.arange(len(self.gt_positions)), cmap='jet')
        plt.scatter(self.template_positions[:, 0], self.template_positions[:, 1], c=np.arange(len(self.template_positions)), cmap='jet', marker='v')

    def plot(self, mode='median', title=None):
        norms = np.linalg.norm(self.waveforms.get_all_templates(mode=mode),  axis=(1, 2))
        errors = np.linalg.norm(self.template_positions - self.gt_positions, axis=1)

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        #ax.set_title(title)
        ax.plot(norms, errors, '.')

        ax = axs[0, 1]
        #ax.set_title(title)
        ax.plot(np.linalg.norm(self.gt_positions, axis=1), errors, '.')
        
        results = {}

        for unit_id in self.waveforms.sorting.unit_ids:
            results[unit_id] = self.spike_positions[self.waveforms.sorting.get_unit_spike_train(unit_id)]
