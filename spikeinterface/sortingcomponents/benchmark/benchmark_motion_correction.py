
import json
import numpy as np
import time
from pathlib import Path

from spikeinterface.core import extract_waveforms
from spikeinterface.extractors import read_mearec
from spikeinterface.preprocessing import bandpass_filter, zscore, common_reference
from spikeinterface.sorters import run_sorter
from spikeinterface.postprocessing import get_template_channel_sparsity
from spikeinterface.widgets import plot_unit_waveforms, plot_gt_performances

from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.widgets import plot_sorting_performance



import matplotlib.pyplot as plt

import MEArec as mr

class BenchmarkMotionCorrectionMearec(BenchmarkBase):
    
    _array_names = ('motion', 'temporal_bins', 'spatial_bins')
    _waveform_names = ('static', 'drifting', 'corrected')
    _sorting_names = ('gt', )

    _array_names_from_parent = ()
    _waveform_names_from_parent = ('static', 'drifting')
    _sorting_names_from_parent = ('gt', 'static', 'drifting')

    def __init__(self, mearec_filename_drifting, mearec_filename_static, 
                motion,
                temporal_bins,
                spatial_bins,
                do_preprocessing=True,
                correct_motion_kwargs={},
                sorter_params={},
                folder=None,
                title='',
                job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}, 
                overwrite=False,
                parent_benchmark=None):

        BenchmarkBase.__init__(self, folder=folder, title=title, overwrite=overwrite, job_kwargs=job_kwargs, parent_benchmark=parent_benchmark)

        self._args.extend([str(mearec_filename_drifting), str(mearec_filename_static), None, None, None ])
        

        self.sorter_params = sorter_params.copy()
        self.mearec_filenames = {}  
        self.keys = ['static', 'drifting', 'corrected']
        self.mearec_filenames['drifting'] = mearec_filename_drifting
        self.mearec_filenames['static'] = mearec_filename_static
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.motion = motion
        self.do_preprocessing = do_preprocessing

        self._recordings = None
        _, self.sortings['gt'] = read_mearec(self.mearec_filenames['static'])
        
        self.correct_motion_kwargs = correct_motion_kwargs.copy()
        self.comparisons = {}

        self._kwargs.update(dict(
                correct_motion_kwargs=self.correct_motion_kwargs,
                sorter_params=self.sorter_params,
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
        for sorter_name, sorter_params in self.sorter_params.items():
            self.run_sorting(sorter_name, sorter_params)
        self.save_to_folder()


    def extract_waveforms(self):
        for key in self.keys:
            if self.parent_benchmark is not None and key in self._waveform_names_from_parent:
                continue
            
            waveforms_folder = self.folder / "waveforms" / key
            self.waveforms[key] = extract_waveforms(self.recordings[key], self.sortings['gt'], waveforms_folder, 'folder', 
                load_if_exists=not self.overwrite, overwrite=self.overwrite, **self.job_kwargs)
    
    def run_sorting(self, sorter_name, sorter_params):
        for key in ('drifting', 'static', 'corrected'):
            if (key, sorter_name) not in self.sortings.keys():
                sorting = run_sorter(sorter_name, self.recordings[key], sorter_name, **sorter_params, delete_output_folder=True)
                self.sortings[key, sorter_name] = sorting
    
    def _compute_templates_similarities(self, metric='cosine', num_channels=30):
        gkey = (metric, num_channels)
        
        if not hasattr(self, '_templates_similarities'):
            self._templates_similarities = {}

        if gkey not in self._templates_similarities:
            import sklearn
            nb_templates = len(self.waveforms['static'].unit_ids)

            sparsity = get_template_channel_sparsity(self.waveforms['static'], 
                        num_channels=num_channels, outputs='index')

            self._templates_similarities[gkey] = {}
            self._templates_similarities[gkey]['norm'] = np.zeros(nb_templates)
            for key in ['drifting', 'corrected']:
                self._templates_similarities[gkey][key] = np.zeros(nb_templates)    
                for unit_ind, unit_id in enumerate(self.waveforms[key].sorting.unit_ids):
                    template = self.waveforms['static'].get_template(unit_id)
                    template = template[:, sparsity[unit_id]].reshape(1, -1)
                    new_template = self.waveforms[key].get_template(unit_id)
                    new_template = new_template[:, sparsity[unit_id]].reshape(1, -1)
                    if metric == 'euclidean':
                        self._templates_similarities[gkey][key][unit_ind] = sklearn.metrics.pairwise_distances(template, new_template)[0]
                    elif metric == 'cosine':
                        self._templates_similarities[gkey][key][unit_ind] = sklearn.metrics.pairwise.cosine_similarity(template, new_template)[0]
                    
                    self._templates_similarities[gkey]['norm'][unit_ind] = np.linalg.norm(template)
        
        return self._templates_similarities[gkey]

    def _compute_snippets_variability(self, metric='cosine', num_channels=30):
        gkey = (metric, num_channels)
        
        if not hasattr(self, '_snippets_variability'):
            self._snippets_variability = {}

        if gkey not in self._snippets_variability:
            import sklearn
            self._snippets_variability[gkey] = {'mean' : {}, 'std' : {}}
            nb_templates = len(self.waveforms['static'].unit_ids)

            sparsity = get_template_channel_sparsity(self.waveforms['static'], 
                        num_channels=num_channels, outputs='index')

            for key in self.keys:
                self._snippets_variability[gkey]['mean'][key] = np.zeros(nb_templates)
                self._snippets_variability[gkey]['std'][key] = np.zeros(nb_templates)

                for unit_ind, unit_id in enumerate(self.waveforms[key].sorting.unit_ids):
                    w = self.waveforms[key].get_waveforms(unit_id)[:, :, sparsity[unit_id]]
                    nb_waveforms = len(w)
                    flat_w = w.reshape(nb_waveforms, -1)
                    template = self.waveforms['static'].get_template(unit_id)
                    template = template[:, sparsity[unit_id]].reshape(1, -1)
                    if metric == 'euclidean':
                        d = sklearn.metrics.pairwise_distances(template, flat_w)[0]
                    elif metric == 'cosine':
                        d = sklearn.metrics.pairwise.cosine_similarity(template, flat_w)[0]
                    self._snippets_variability[gkey]['mean'][key][unit_ind] = d.mean()
                    self._snippets_variability[gkey]['std'][key][unit_ind] = d.std()
        return self._snippets_variability[gkey]
    
    def compare_snippets_variability(self, metric='cosine', num_channels=30):
        
        results = self._compute_snippets_variability(metric, num_channels)

        fig, axes = plt.subplots(2  , 3, figsize=(15, 10))

        colors = 0
        labels = []
        for key in self.keys:
            axes[0, 0].violinplot(results['mean'][key], [colors], showmeans=True)
            colors += 1

        _simpleaxis(axes[0, 0])
        axes[0, 0].set_xticks(np.arange(len(self.keys)), self.keys)
        if metric == 'euclidean':
            axes[0, 0].set_ylabel(r'mean($\| snippets - template\|_2)$')
        elif metric == 'cosine':
            axes[0, 0].set_ylabel('mean(cosine(snippets, template))')
        
        colors = 0
        labels = []
        for key in self.keys:
            axes[0, 1].violinplot(results['std'][key], [colors], showmeans=True)
            colors += 1

        _simpleaxis(axes[0, 1])
        axes[0, 1].set_xticks(np.arange(len(self.keys)), self.keys)
        if metric == 'euclidean':
            axes[0, 1].set_ylabel(r'std($\| snippets - template\|_2)$')
        elif metric == 'cosine':
            axes[0, 1].set_ylabel('std(cosine(snippets, template))')

        distances = self._compute_templates_similarities(metric, num_channels)

        axes[0, 2].scatter(distances['drifting'], distances['corrected'], c='k', alpha=0.5)
        xmin, xmax = axes[0, 2].get_xlim()
        axes[0, 2].plot([xmin, xmax], [xmin, xmax], 'k--')
        _simpleaxis(axes[0, 2])
        if metric == 'euclidean':
            axes[0, 2].set_xlabel(r'$\|drift - static\|_2$')
            axes[0, 2].set_ylabel(r'$\|corrected - static\|_2$')
        elif metric == 'cosine':
            axes[0, 2].set_xlabel(r'$cosine(drift, static)$')
            axes[0, 2].set_ylabel(r'$cosine(corrected, static)$')

        import MEArec as mr
        recgen = mr.load_recordings(self.mearec_filenames['static'])
        nb_templates, nb_versions, _ = recgen.template_locations.shape
        template_positions = recgen.template_locations[:, nb_versions//2, 1:3]
        distances_to_center = template_positions[:, 1]

        differences = {}
        differences['corrected'] = results['mean']['corrected']/results['mean']['static']
        differences['drifting'] = results['mean']['drifting']/results['mean']['static']
        axes[1, 0].scatter(distances['norm'], differences['corrected'], color='C2')
        axes[1, 0].scatter(distances['norm'], differences['drifting'], color='C1')
        if metric == 'euclidean':
            axes[1, 0].set_ylabel(r'$\Delta \|~\|_2$ (% static)')
        elif metric == 'cosine':
            axes[1, 0].set_ylabel(r'$\Delta cosine$  (% static)')
        axes[1, 0].set_xlabel('template norm')
        xmin, xmax = axes[1, 0].get_xlim()
        axes[1, 0].plot([xmin, xmax], [0, 0], 'k--')
        _simpleaxis(axes[1, 0])

        axes[1, 1].scatter(distances_to_center, differences['drifting'], color='C1')
        axes[1, 1].scatter(distances_to_center, differences['corrected'], color='C2')
        if metric == 'euclidean':
            axes[1, 1].set_ylabel(r'$\Delta \|~\|_2$  (% static)')
        elif metric == 'cosine':
            axes[1, 1].set_ylabel(r'$\Delta cosine$  (% static)')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('depth (um)')
        xmin, xmax = axes[1, 1].get_xlim()
        axes[1, 1].plot([xmin, xmax], [0, 0], 'k--')
        _simpleaxis(axes[1, 1])

        colors = 0
        labels = []
        for key in ['drifting', 'corrected']:
            axes[1, 2].bar([colors], [differences[key].mean()], color=f'C{colors+1}')
            colors += 1

        _simpleaxis(axes[1, 2])
        axes[1, 2].set_xticks(np.arange(2), ['drifting', 'corrected'])
        if metric == 'euclidean':
            axes[1, 2].set_ylabel(r'$\Delta \|~\|_2$  (% static)')
        elif metric == 'cosine':
            axes[1, 2].set_ylabel(r'$\Delta cosine$  (% static)')

    def _get_residuals(self, key, time_range):
        gkey = key, time_range
        
        if not hasattr(self, '_residuals'):
            self._residuals = {}
        
        fr = int(self.recordings['static'].get_sampling_frequency())
        duration = int(self.recordings['static'].get_total_duration())

        if time_range is None:
            t_start = 0
            t_stop = duration
        else:
            t_start, t_stop = time_range

        if gkey not in self._residuals:
            difference = ResidualRecording(self.recordings['static'], self.recordings[key])
            self._residuals[gkey] = np.zeros((self.recordings['static'].get_num_channels(), 0))
            
            for i in np.arange(t_start*fr, t_stop*fr, fr):
                data = np.linalg.norm(difference.get_traces(start_frame=i, end_frame=i+fr), axis=0)/np.sqrt(fr)
                self._residuals[gkey] = np.hstack((self._residuals[gkey], data[:,np.newaxis]))
        
        return self._residuals[gkey], (t_start, t_stop)

    def compare_residuals(self, time_range=None):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        residuals = {}

        for key in ['drifting', 'corrected']:
            residuals[key], (t_start, t_stop) = self._get_residuals(key, time_range)

        time_axis = np.arange(t_start, t_stop)
        axes[0 ,0].plot(time_axis, residuals['drifting'].mean(0), label=r'$|S_{drifting} - S_{static}|$')
        axes[0 ,0].plot(time_axis, residuals['corrected'].mean(0), label=r'$|S_{corrected} - S_{static}|$')
        axes[0 ,0].legend()
        axes[0, 0].set_xlabel('time (s)')
        axes[0, 0].set_ylabel('mean residual')
        _simpleaxis(axes[0, 0])

        channel_positions = self.recordings['static'].get_channel_locations()
        distances_to_center = channel_positions[:, 1]
        idx = np.argsort(distances_to_center)

        axes[0, 1].plot(distances_to_center[idx], residuals['drifting'].mean(1)[idx], label=r'$|S_{drift} - S_{static}|$')
        axes[0, 1].plot(distances_to_center[idx], residuals['corrected'].mean(1)[idx], label=r'$|S_{corrected} - S_{static}|$')
        axes[0, 1].legend()
        axes[0 ,1].set_xlabel('depth (um)')
        axes[0, 1].set_ylabel('mean residual')
        _simpleaxis(axes[0, 1])

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        peaks = detect_peaks(self.recordings['static'], method='by_channel', **self.job_kwargs)

        fr = int(self.recordings['static'].get_sampling_frequency())
        duration = int(self.recordings['static'].get_total_duration())
        mask = (peaks['sample_ind'] >= t_start*fr) & (peaks['sample_ind'] <= t_stop*fr)

        _, counts = np.unique(peaks['channel_ind'][mask], return_counts=True)
        counts = counts.astype(np.float64) / (t_stop - t_start)

        axes[1, 0].plot(distances_to_center[idx],(fr*residuals['drifting'].mean(1)/counts)[idx], label='drifting')
        axes[1, 0].plot(distances_to_center[idx],(fr*residuals['corrected'].mean(1)/counts)[idx], label='corrected')
        axes[1, 0].set_ylabel('mean residual / rate')
        axes[1, 0].set_xlabel('depth of the channel [um]')
        axes[1, 0].legend()
        _simpleaxis(axes[1, 0])

        axes[1, 1].scatter(counts, residuals['drifting'].mean(1), label='drifting')
        axes[1, 1].scatter(counts, residuals['corrected'].mean(1), label='corrected')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('rate per channel (Hz)')
        axes[1, 1].set_ylabel('Mean residual')
        _simpleaxis(axes[1,1])

    def compare_waveforms(self, unit_id, num_channels=20):
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))

        sparsity = get_template_channel_sparsity(self.waveforms['static'], 
                        num_channels=num_channels)

        for count, key in enumerate(self.keys):

            plot_unit_waveforms(self.waveforms[key], unit_ids=[unit_id], ax=axes[count], 
                unit_colors={unit_id : 'k'}, same_axis=True, alpha_waveforms=0.05, sparsity=sparsity)
            axes[count].set_title(f'unit {unit_id} {key}')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            _simpleaxis(axes[count])
            axes[count].spines['bottom'].set_visible(False)
            axes[count].spines['left'].set_visible(False)

def plot_snippet_comparisons(benchmarks, metric='cosine', num_channels=30):

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2)

    ax = fig.add_subplot(gs[0:2, 0])
    for count, bench in enumerate(benchmarks):
        distances = bench._compute_templates_similarities(metric, num_channels)
        ax.scatter(distances['drifting'], distances['corrected'], c=f'C{count}', alpha=0.5, label=bench.title)

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


    import MEArec as mr
    recgen = mr.load_recordings(benchmarks[0].mearec_filenames['static'])
    nb_templates, nb_versions, _ = recgen.template_locations.shape
    template_positions = recgen.template_locations[:, nb_versions//2, 1:3]
    distances_to_center = template_positions[:, 1]

    ax_1 = fig.add_subplot(gs[0, 1])
    ax_2 = fig.add_subplot(gs[1, 1])
    ax_3 = fig.add_subplot(gs[2:, 1])
    ax_4 = fig.add_subplot(gs[2:, 0])

    for count, bench in enumerate(benchmarks):
        results = bench._compute_snippets_variability(metric, num_channels)
        m_differences = results['mean']['corrected']/results['mean']['static']
        s_differences = results['std']['corrected']/results['std']['static']
        ax_3.bar([count], [m_differences.mean()], yerr=[m_differences.std()], color=f'C{count}')
        ax_4.bar([count], [s_differences.mean()], yerr=[s_differences.std()], color=f'C{count}')
        idx = np.argsort(distances_to_center)
        ax_1.plot(distances_to_center[idx], m_differences[idx], color=f'C{count}')
        ax_2.plot(distances_to_center[idx], s_differences[idx], color=f'C{count}')

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

colors = {'static' : 'C0', 'drifting' : 'C1', 'corrected' : 'C2'}

def plot_sortings_comparisons(benchmarks, sorter_name, mode='swarmplot'):

    all_comparisons = []

    for count, bench in enumerate(benchmarks):
        all_comparisons.append({})
        for key, sorting in bench.sortings.items():
            if key != 'gt' and key[1] == sorter_name:
                if key not in bench.comparisons:
                    bench.comparisons[key] = GroundTruthComparison(bench.sortings['gt'], bench.sortings[key], exhaustive_gt=True)
                all_comparisons[-1][key] = bench.comparisons[key]

    if mode == 'swarmplot':
        fig, axes = plt.subplots(len(benchmarks), 3, figsize=(15, 5), squeeze=False)
        for count, comparisons in enumerate(all_comparisons):
            for key, comp in comparisons.items():
                if key[1] != sorter_name:
                    continue
                idx = ('static', 'drifting', 'corrected').index(key[0])
                ax = axes[count, idx]
                plot_gt_performances(comp, ax=ax)
                if idx > 0:
                    ax.set_yticks([])
                    ax.set_ylabel('')
                ax.set_title(key)
                _simpleaxis(ax)

    elif mode == 'snr':
        fig, axes = plt.subplots(len(benchmarks), 2, figsize=(15, 5), squeeze=False)
        metrics = compute_quality_metrics(benchmarks[0].waveforms['static'], metric_names=['snr'], load_if_exists=True)
        for count, comparisons in enumerate(all_comparisons):
            for key, comp in comparisons.items():
                if key[1] != sorter_name:
                    continue
                ax = axes[count, 0]
                plot_sorting_performance(comp, metrics, performance_name='accuracy', metric_name='snr', ax=ax, color=colors[key[0]])
                _simpleaxis(ax)
            

            ax = axes[count, 1]
            perf_static = comparisons[('static', sorter_name)].get_performance()['accuracy']
            perf_drifting = comparisons[('drifting', sorter_name)].get_performance()['accuracy']
            perf_corrected = comparisons[('corrected', sorter_name)].get_performance()['accuracy']
            ax.plot(metrics['snr'], perf_static - perf_drifting, marker='o', markersize=5, ls='', color=colors['drifting'], label='drifting')
            ax.plot(metrics['snr'], perf_static - perf_corrected, marker='o', markersize=5, ls='', color=colors['corrected'], label='corrected')
            ax.set_xlabel('snr')
            ax.legend()
            ax.set_ylabel(r'$\Delta$ accuracy')
    elif mode == 'scatter':
        fig, axes = plt.subplots(len(benchmarks), 1, figsize=(15, 5), squeeze=False)
        metrics = compute_quality_metrics(benchmarks[0].waveforms['static'], metric_names=['snr'], load_if_exists=True)
        for count, comparisons in enumerate(all_comparisons):
            
            ax = axes[count, 0]
            perf_static = comparisons[('static', sorter_name)].get_performance()['accuracy']
            perf_drifting = comparisons[('drifting', sorter_name)].get_performance()['accuracy']
            perf_corrected = comparisons[('corrected', sorter_name)].get_performance()['accuracy']
            ax.plot(perf_static, perf_drifting, marker='o', markersize=5, ls='', color=colors['drifting'], label='drifting')
            ax.plot(perf_static, perf_corrected, marker='o', markersize=5, ls='', color=colors['corrected'], label='corrected')
            ax.plot([0, 1], [0, 1], 'k')
            ax.set_xlim(0.5, 1)
            ax.set_ylim(0.5, 1)
            ax.legend()
            ax.set_xlabel('accuracy static')
            ax.set_xlabel('accuracy recording')
