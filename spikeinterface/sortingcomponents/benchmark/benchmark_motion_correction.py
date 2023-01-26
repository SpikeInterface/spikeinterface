

import numpy as np
import pandas as pd

from pathlib import Path
import shutil

from spikeinterface.core import extract_waveforms, precompute_sparsity, WaveformExtractor


from spikeinterface.extractors import read_mearec
from spikeinterface.preprocessing import bandpass_filter, zscore, common_reference
from spikeinterface.sorters import run_sorter
from spikeinterface.widgets import plot_unit_waveforms, plot_gt_performances

from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.widgets import plot_sorting_performance


import sklearn

import matplotlib.pyplot as plt

import MEArec as mr

class BenchmarkMotionCorrectionMearec(BenchmarkBase):
    
    _array_names = ('motion', 'temporal_bins', 'spatial_bins')
    _waveform_names = ('static', 'drifting', 'corrected')
    _sorting_names = ()

    _array_names_from_parent = ()
    _waveform_names_from_parent = ('static', 'drifting')
    _sorting_names_from_parent = ('static', 'drifting')

    def __init__(self, mearec_filename_drifting, mearec_filename_static, 
                motion,
                temporal_bins,
                spatial_bins,
                do_preprocessing=True,
                correct_motion_kwargs={},
                sparse_kwargs=dict( method="radius", peak_sign="neg", radius_um=100.,),
                sorter_cases={},
                folder=None,
                title='',
                job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}, 
                overwrite=False,
                parent_benchmark=None):

        BenchmarkBase.__init__(self, folder=folder, title=title, overwrite=overwrite, job_kwargs=job_kwargs,
                               parent_benchmark=parent_benchmark)

        self._args.extend([str(mearec_filename_drifting), str(mearec_filename_static), None, None, None ])
        

        self.sorter_cases = sorter_cases.copy()
        self.mearec_filenames = {}  
        self.keys = ['static', 'drifting', 'corrected']
        self.mearec_filenames['drifting'] = mearec_filename_drifting
        self.mearec_filenames['static'] = mearec_filename_static
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.motion = motion
        self.do_preprocessing = do_preprocessing

        self._recordings = None
        _, self.sorting_gt = read_mearec(self.mearec_filenames['static'])
        
        self.correct_motion_kwargs = correct_motion_kwargs.copy()
        self.sparse_kwargs = sparse_kwargs.copy()
        self.comparisons = {}
        self.accuracies = {}

        self._kwargs.update(dict(
                correct_motion_kwargs=self.correct_motion_kwargs,
                sorter_cases=self.sorter_cases,
                do_preprocessing=do_preprocessing,
                sparse_kwargs=sparse_kwargs,
            )
        )

    @property
    def recordings(self):
        if self._recordings is None:
            self._recordings = {}
            for key in ('drifting', 'static',):
                rec, _  = read_mearec(self.mearec_filenames[key])
                if self.do_preprocessing:
                    rec = bandpass_filter(rec)
                    rec = common_reference(rec)
                    rec = zscore(rec)
                self._recordings[key] = rec

            rec = self._recordings['drifting']
            self._recordings['corrected'] = CorrectMotionRecording(rec, self.motion, 
                        self.temporal_bins, self.spatial_bins, **self.correct_motion_kwargs)
        return self._recordings

    def run(self):
        self.extract_waveforms()
        self.save_to_folder()
        #self.run_sorters()
        #self.save_to_folder()


    def extract_waveforms(self):

        # the sparsity is estimated on the static recording and propagated to all of then
        if self.parent_benchmark is None:
            sparsity = precompute_sparsity(self.recordings['static'], self.sorting_gt,
                                       ms_before=2., ms_after=3., num_spikes_for_sparsity=200., unit_batch_size=10000,
                                       **self.sparse_kwargs, **self.job_kwargs)
        else:
            sparsity = self.waveforms['static'].sparsity

        for key in self.keys:
            if self.parent_benchmark is not None and key in self._waveform_names_from_parent:
                continue
            
            waveforms_folder = self.folder / "waveforms" / key
            we = WaveformExtractor.create(self.recordings[key], self.sorting_gt, waveforms_folder, mode='folder',
                                          sparsity=sparsity)
            we.set_params(ms_before=2., ms_after=3., max_spikes_per_unit=500., return_scaled=True)
            we.run_extract_waveforms(seed=22051977, **self.job_kwargs)
            self.waveforms[key] = we


    def run_sorters(self):
        for case in self.sorter_cases:
            label = case['label']
            print('run sorter', label)
            sorter_name = case['sorter_name']
            sorter_params = case['sorter_params']
            recording = self.recordings[case['recording']]
            output_folder = self.folder / f'tmp_sortings_{label}'
            sorting = run_sorter(sorter_name, recording, output_folder, **sorter_params, delete_output_folder=True)
            self.sortings[label] = sorting


    def compute_distances_to_static(self, force=False):
        if hasattr(self, 'distances') and not force:
            return self.distances

        self.distances = {}

        n = len(self.waveforms['static'].unit_ids)

        sparsity = self.waveforms['static'].sparsity

        ref_templates = self.waveforms['static'].get_all_templates()
        
        # for key in ['drifting', 'corrected']:
        for key in self.keys:
            dist = self.distances[key] = {
                                        'norm_static' : np.zeros(n),
                                        'template_euclidean' : np.zeros(n),
                                        'template_cosine' : np.zeros(n),
                                        'wf_euclidean_mean' : np.zeros(n),
                                        'wf_euclidean_std' : np.zeros(n),
                                        'wf_cosine_mean' : np.zeros(n),
                                        'wf_cosine_std' : np.zeros(n),
                                        }
            templates = self.waveforms[key].get_all_templates()
            for unit_ind, unit_id in enumerate(self.waveforms[key].sorting.unit_ids):
                mask = sparsity.mask[unit_ind, :]
                ref_template = ref_templates[unit_ind][:, mask].reshape(1, -1)
                template = templates[unit_ind][:, mask].reshape(1, -1)

                # this is already sparse
                # ref_wfs = self.waveforms['static'].get_waveforms(unit_id)
                # ref_wfs = ref_wfs.reshape(ref_wfs.shape[0], -1)
                wfs = self.waveforms[key].get_waveforms(unit_id)
                wfs = wfs.reshape(wfs.shape[0], -1)

                dist['norm_static'][unit_ind] = np.linalg.norm(ref_template)
                dist['template_euclidean'][unit_ind] = sklearn.metrics.pairwise_distances(ref_template, template)[0]
                dist['template_cosine'][unit_ind] = sklearn.metrics.pairwise.cosine_similarity(ref_template, template)[0]

                d = sklearn.metrics.pairwise_distances(ref_template, wfs)[0]
                dist['wf_euclidean_mean'][unit_ind] = d.mean()
                dist['wf_euclidean_std'][unit_ind] = d.std()

                d = sklearn.metrics.pairwise.cosine_similarity(ref_template, wfs)[0]
                dist['wf_cosine_mean'][unit_ind] = d.mean()
                dist['wf_cosine_std'][unit_ind] = d.std()


        return self.distances


    # def _get_residuals(self, key, time_range):
    #     gkey = key, time_range

    #     if not hasattr(self, '_residuals'):
    #         self._residuals = {}
        
    #     fr = int(self.recordings['static'].get_sampling_frequency())
    #     duration = int(self.recordings['static'].get_total_duration())

    #     if time_range is None:
    #         t_start = 0
    #         t_stop = duration
    #     else:
    #         t_start, t_stop = time_range

    #     if gkey not in self._residuals:
    #         difference = ResidualRecording(self.recordings['static'], self.recordings[key])
    #         self._residuals[gkey] = np.zeros((self.recordings['static'].get_num_channels(), 0))
            
    #         for i in np.arange(t_start*fr, t_stop*fr, fr):
    #             data = np.linalg.norm(difference.get_traces(start_frame=i, end_frame=i+fr), axis=0)/np.sqrt(fr)
    #             self._residuals[gkey] = np.hstack((self._residuals[gkey], data[:,np.newaxis]))
        
    #     return self._residuals[gkey], (t_start, t_stop)

    # def compare_residuals(self, time_range=None):

    #     fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    #     residuals = {}

    #     for key in ['drifting', 'corrected']:
    #         residuals[key], (t_start, t_stop) = self._get_residuals(key, time_range)

    #     time_axis = np.arange(t_start, t_stop)
    #     axes[0 ,0].plot(time_axis, residuals['drifting'].mean(0), label=r'$|S_{drifting} - S_{static}|$')
    #     axes[0 ,0].plot(time_axis, residuals['corrected'].mean(0), label=r'$|S_{corrected} - S_{static}|$')
    #     axes[0 ,0].legend()
    #     axes[0, 0].set_xlabel('time (s)')
    #     axes[0, 0].set_ylabel('mean residual')
    #     _simpleaxis(axes[0, 0])

    #     channel_positions = self.recordings['static'].get_channel_locations()
    #     distances_to_center = channel_positions[:, 1]
    #     idx = np.argsort(distances_to_center)

    #     axes[0, 1].plot(distances_to_center[idx], residuals['drifting'].mean(1)[idx], label=r'$|S_{drift} - S_{static}|$')
    #     axes[0, 1].plot(distances_to_center[idx], residuals['corrected'].mean(1)[idx], label=r'$|S_{corrected} - S_{static}|$')
    #     axes[0, 1].legend()
    #     axes[0 ,1].set_xlabel('depth (um)')
    #     axes[0, 1].set_ylabel('mean residual')
    #     _simpleaxis(axes[0, 1])

    #     from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    #     peaks = detect_peaks(self.recordings['static'], method='by_channel', **self.job_kwargs)

    #     fr = int(self.recordings['static'].get_sampling_frequency())
    #     duration = int(self.recordings['static'].get_total_duration())
    #     mask = (peaks['sample_ind'] >= t_start*fr) & (peaks['sample_ind'] <= t_stop*fr)

    #     _, counts = np.unique(peaks['channel_ind'][mask], return_counts=True)
    #     counts = counts.astype(np.float64) / (t_stop - t_start)

    #     axes[1, 0].plot(distances_to_center[idx],(fr*residuals['drifting'].mean(1)/counts)[idx], label='drifting')
    #     axes[1, 0].plot(distances_to_center[idx],(fr*residuals['corrected'].mean(1)/counts)[idx], label='corrected')
    #     axes[1, 0].set_ylabel('mean residual / rate')
    #     axes[1, 0].set_xlabel('depth of the channel [um]')
    #     axes[1, 0].legend()
    #     _simpleaxis(axes[1, 0])

    #     axes[1, 1].scatter(counts, residuals['drifting'].mean(1), label='drifting')
    #     axes[1, 1].scatter(counts, residuals['corrected'].mean(1), label='corrected')
    #     axes[1, 1].legend()
    #     axes[1, 1].set_xlabel('rate per channel (Hz)')
    #     axes[1, 1].set_ylabel('Mean residual')
    #     _simpleaxis(axes[1,1])

    # def compare_waveforms(self, unit_id, num_channels=20):
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    #     sparsity = compute_sparsity(self.waveforms['static'],  method="best_channels", num_channels=num_channels)
    #     for count, key in enumerate(self.keys):

    #         plot_unit_waveforms(self.waveforms[key], unit_ids=[unit_id], ax=axes[count], 
    #             unit_colors={unit_id : 'k'}, same_axis=True, alpha_waveforms=0.05, sparsity=sparsity)
    #         axes[count].set_title(f'unit {unit_id} {key}')
    #         axes[count].set_xticks([])
    #         axes[count].set_yticks([])
    #         _simpleaxis(axes[count])
    #         axes[count].spines['bottom'].set_visible(False)
    #         axes[count].spines['left'].set_visible(False)
    
    def compute_accuracies(self):
        for case in self.sorter_cases:
            label = case['label']
            sorting = self.sortings[label]
            if label not in self.comparisons:
                comp = GroundTruthComparison(self.sorting_gt, sorting, exhaustive_gt=True)
                self.comparisons[label] = comp
                self.accuracies[label] = comp.get_performance()['accuracy'].values

    def plot_sortings_accuracy(self, mode='ordered_accuracy', figsize=(15, 5)):

        self.compute_accuracies()

        n = len(self.sorter_cases)

        if mode == 'ordered_accuracy':
            fig, ax = plt.subplots(figsize=figsize)

            order = None
            for case in self.sorter_cases:
                label = case['label']                
                comp = self.comparisons[label]
                acc = self.accuracies[label]
                order = np.argsort(acc)[::-1]
                acc = acc[order]
                ax.plot(acc, label=label)
            ax.legend()
            ax.set_ylabel('accuracy')
            ax.set_xlabel('units ordered by accuracy')
        
        elif mode == 'depth_snr':
            fig, axs = plt.subplots(nrows=n, figsize=figsize, sharey=True, sharex=True)

            gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filenames['drifting'], time_vector=np.array([0., 1.]))
            depth = gt_unit_positions[0, :]

            chan_locations = self.recordings['drifting'].get_channel_locations()

            metrics = compute_quality_metrics(self.waveforms['static'], metric_names=['snr'], load_if_exists=True)
            snr = metrics['snr'].values

            for i, case in enumerate(self.sorter_cases):
                ax = axs[i]
                label = case['label']
                acc = self.accuracies[label]
                s = ax.scatter(depth, snr, c=acc)
                s.set_clim(0., 1.)
                ax.set_title(label)
                ax.axvline(np.min(chan_locations[:, 1]), ls='--', color='k')
                ax.axvline(np.max(chan_locations[:, 1]), ls='--', color='k')
            ax.set_xlabel('depth')
            ax.set_ylabel('snr')


        elif mode == 'snr':
            fig, ax = plt.subplots(figsize=figsize)

            metrics = compute_quality_metrics(self.waveforms['static'], metric_names=['snr'], load_if_exists=True)
            snr = metrics['snr'].values

            for i, case in enumerate(self.sorter_cases):
                label = case['label']
                acc = self.accuracies[label]
                ax.scatter(snr, acc, label=label)
            ax.set_xlabel('snr')
            ax.set_ylabel('accuracy')

            ax.legend()


        elif mode == 'depth':
            fig, ax = plt.subplots(figsize=figsize)

            gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filenames['drifting'], time_vector=np.array([0., 1.]))
            depth = gt_unit_positions[0, :]

            chan_locations = self.recordings['drifting'].get_channel_locations()

            for i, case in enumerate(self.sorter_cases):
                label = case['label']
                acc = self.accuracies[label]
                ax.scatter(depth, acc, label=label)
            ax.axvline(np.min(chan_locations[:, 1]), ls='--', color='k')
            ax.axvline(np.max(chan_locations[:, 1]), ls='--', color='k')
            ax.legend()
            ax.set_xlabel('depth')
            ax.set_ylabel('accuracy')



def plot_distances_to_static(benchmarks, metric='cosine'):

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2)

    ax = fig.add_subplot(gs[0:2, 0])
    for count, bench in enumerate(benchmarks):

        print(bench)
        distances = bench.compute_distances_to_static(force=False)
        print(distances.keys())
        ax.scatter(distances['drifting'][f'template_{metric}'], distances['corrected'][f'template_{metric}'], c=f'C{count}', alpha=0.5, label=bench.title)

    ax.legend()


    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], 'k--')
    _simpleaxis(ax)
    if metric == 'euclidean':
        ax.set_xlabel(r'$\|drift - static\|_2$')
        ax.set_ylabel(r'$\|corrected - static\|_2$')
    elif metric == 'cosine':
        ax.set_xlabel(r'$cosine(drift, static)$')
        ax.set_ylabel(r'$cosine(corrected, static)$')


    recgen = mr.load_recordings(benchmarks[0].mearec_filenames['static'])
    nb_templates, nb_versions, _ = recgen.template_locations.shape
    template_positions = recgen.template_locations[:, nb_versions//2, 1:3]
    distances_to_center = template_positions[:, 1]

    ax_1 = fig.add_subplot(gs[0, 1])
    ax_2 = fig.add_subplot(gs[1, 1])
    ax_3 = fig.add_subplot(gs[2:, 1])
    ax_4 = fig.add_subplot(gs[2:, 0])

    for count, bench in enumerate(benchmarks):

        # results = bench._compute_snippets_variability(metric=metric, num_channels=num_channels)
        distances = bench.compute_distances_to_static(force=False)

        m_differences = distances['corrected'][f'wf_{metric}_mean']/distances['static'][f'wf_{metric}_mean']
        s_differences = distances['corrected'][f'wf_{metric}_std']/distances['static'][f'wf_{metric}_std']

        ax_3.bar([count], [m_differences.mean()], yerr=[m_differences.std()], color=f'C{count}')
        ax_4.bar([count], [s_differences.mean()], yerr=[s_differences.std()], color=f'C{count}')
        idx = np.argsort(distances_to_center)
        ax_1.scatter(distances_to_center[idx], m_differences[idx], color=f'C{count}')
        ax_2.scatter(distances_to_center[idx], s_differences[idx], color=f'C{count}')

    for a in [ax_1, ax_2, ax_3, ax_4]:
        _simpleaxis(a)
    
    if metric == 'euclidean':
        ax_1.set_ylabel(r'$\Delta mean(\|~\|_2)$  (% static)')
        ax_2.set_ylabel(r'$\Delta std(\|~\|_2)$  (% static)')
        ax_3.set_ylabel(r'$\Delta mean(\|~\|_2)$  (% static)')
        ax_4.set_ylabel(r'$\Delta std(\|~\|_2)$  (% static)')
    elif metric == 'cosine':
        ax_1.set_ylabel(r'$\Delta mean(cosine)$  (% static)')
        ax_2.set_ylabel(r'$\Delta std(cosine)$  (% static)')
        ax_3.set_ylabel(r'$\Delta mean(cosine)$  (% static)')
        ax_4.set_ylabel(r'$\Delta std(cosine)$  (% static)')
    ax_3.set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
    ax_4.set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
    xmin, xmax = ax_3.get_xlim()
    ax_3.plot([xmin, xmax], [1, 1], 'k--')
    ax_4.plot([xmin, xmax], [1, 1], 'k--')
    ax_1.set_xticks([])
    ax_2.set_xlabel('depth (um)')

    xmin, xmax = ax_1.get_xlim()
    ax_1.plot([xmin, xmax], [1, 1], 'k--')
    ax_2.plot([xmin, xmax], [1, 1], 'k--')
    plt.tight_layout()


def plot_residuals_comparisons(benchmarks):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench._get_residuals('corrected', None)
        time_axis = np.arange(t_start, t_stop)
        axes[0].plot(time_axis, residuals.mean(0), label=bench.title)
    axes[0].legend()
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel(r'$|S_{corrected} - S_{static}|$')
    _simpleaxis(axes[0])

    channel_positions = benchmarks[0].recordings['static'].get_channel_locations()
    distances_to_center = channel_positions[:, 1]
    idx = np.argsort(distances_to_center)

    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench._get_residuals('corrected', None)
        time_axis = np.arange(t_start, t_stop)
        axes[1].plot(distances_to_center[idx], residuals.mean(1)[idx], label=bench.title, lw=2, c=f'C{count}')
        axes[1].fill_between(distances_to_center[idx], residuals.mean(1)[idx]-residuals.std(1)[idx], 
                    residuals.mean(1)[idx]+residuals.std(1)[idx], color=f'C{count}', alpha=0.25)
    #axes[1].legend()
    axes[1].set_xlabel('depth (um)')
    #axes[1].set_ylabel(r'$|S_{corrected} - S_{static}|$')
    _simpleaxis(axes[1])

    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench._get_residuals('corrected', None)
        axes[2].bar([count], [residuals.mean()], yerr=[residuals.std()], color=f'C{count}')

    _simpleaxis(axes[2])
    axes[2].set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
    #axes[2].set_ylabel(r'$|S_{corrected} - S_{static}|$')


from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
class ResidualRecording(BasePreprocessor):
    name = 'residual_recording'
    def __init__(self, recording_1, recording_2):
        assert recording_1.get_num_segments() == recording_2.get_num_segments()
        BasePreprocessor.__init__(self, recording_1)

        for parent_recording_segment_1, parent_recording_segment_2 in zip(recording_1._recording_segments, recording_2._recording_segments):
            rec_segment = DifferenceRecordingSegment(parent_recording_segment_1, parent_recording_segment_2)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording_1=recording_1.to_dict(), recording_2=recording_2.to_dict())


class DifferenceRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment_1, parent_recording_segment_2):
        BasePreprocessorSegment.__init__(self, parent_recording_segment_1)
        self.parent_recording_segment_1 = parent_recording_segment_1
        self.parent_recording_segment_2 = parent_recording_segment_2

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces_1 = self.parent_recording_segment_1.get_traces(start_frame, end_frame, channel_indices)
        traces_2 = self.parent_recording_segment_2.get_traces(start_frame, end_frame, channel_indices)

        return traces_2 - traces_1

# colors = {'static' : 'C0', 'drifting' : 'C1', 'corrected' : 'C2'}

