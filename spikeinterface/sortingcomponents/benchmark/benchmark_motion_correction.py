
import json
import numpy as np
import time
from pathlib import Path


from spikeinterface.core import get_noise_levels
from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
from spikeinterface.sortingcomponents.motion_correction import correct_motion_on_peaks

from spikeinterface.core import extract_waveforms, load_waveforms

from spikeinterface.widgets import plot_probe_map

import scipy.interpolate

import matplotlib.pyplot as plt

import MEArec as mr

class BenchmarkMotionCorrectionMearec:
    
    def __init__(self, mearec_filename_drifting, mearec_filename_static, 
                motion,
                temporal_bins, 
                spatial_bins,
                title='',
                correct_motion_kwargs={},
                output_folder=None,
                job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}, 
                overwrite=False):
        
        self.mearec_filenames = {}  
        self.keys = ['static', 'drifting', 'corrected']
        self.mearec_filenames['drifting'] = mearec_filename_drifting
        self.mearec_filenames['static'] = mearec_filename_static
        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins

        self._recordings = None
        self.sortings = {}
        for key in ['drifting', 'static']:
            _, self.sortings[key] = read_mearec(self.mearec_filenames[key])

        self.title = title
        self.job_kwargs = job_kwargs
        self.correct_motion_kwargs = correct_motion_kwargs
        self.overwrite = overwrite
        self.output_folder = output_folder

    @property
    def recordings(self):
        if self._recordings is None:
            self._recordings = {}
            for key in ['drifting', 'static']:
                self._recordings[key], _ = read_mearec(self.mearec_filenames[key])

            self._recordings['corrected'] = CorrectMotionRecording(self._recordings['drifting'], self.motion, 
                        self.temporal_bins, self.spatial_bins, **self.correct_motion_kwargs)
            self.sortings['corrected'] = self.sortings['static']
        return self._recordings

    def extract_waveforms(self):

        self.waveforms = {}
        for key in self.keys:
            if self.output_folder is None:
                mode = 'memory'
                waveforms_folder = None
            else:
                mode = 'folder'
                waveforms_folder = self.output_folder / "waveforms" / key
            self.waveforms[key] = extract_waveforms(self.recordings[key], self.sortings[key], waveforms_folder, mode, 
                load_if_exists=not self.overwrite, overwrite=self.overwrite, **self.job_kwargs)

    _array_names = ('motion', 'temporal_bins', 'spatial_bins')
    _dict_kwargs_names = ('job_kwargs', 'correct_motion_kwargs', 'mearec_filenames')

    def save_to_folder(self, folder):

        if folder.exists() and self.overwrite:
            import shutil
            shutil.rmtree(folder)
            folder.mkdir(parents=True)
        elif not folder.exists():
            folder.mkdir(parents=True)

        folder = Path(folder)

        info = {
            'mearec_filename_static': str(self.mearec_filenames['static']),
            'mearec_filename_drifting': str(self.mearec_filenames['drifting']),
            'title': self.title,
        }
        (folder / 'info.json').write_text(json.dumps(info, indent=4), encoding='utf8')

        for name in self._array_names:
            value = getattr(self, name)
            if value is not None:
                np.save(folder / f'{name}.npy', value)
        
        for name in self._dict_kwargs_names:
            d = getattr(self, name)
            file = folder / f'{name}.json'
            if d is not None:
                file.write_text(json.dumps(d, indent=4), encoding='utf8')

        #run_times_filename = folder / 'run_times.json'
        #run_times_filename.write_text(json.dumps(self.run_times, indent=4),encoding='utf8')

    @classmethod
    def load_from_folder(cls, folder):
        folder = Path(folder)
        assert folder.exists()

        with open(folder / 'info.json', 'r') as f:
            info = json.load(f)
        title = info['title']

        dict_kwargs = dict()
        for name in cls._dict_kwargs_names:
            filename = folder / f'{name}.json' 
            if filename.exists():
                with open(filename, 'r') as f:
                    d = json.load(f)
            else:
                d = None
            dict_kwargs[name] = d

        mearec_filenames = dict_kwargs.pop('mearec_filenames')
        bench = cls(mearec_filenames['drifting'], mearec_filenames['static'], 
            None,
            None,
            None, 
            output_folder=folder, title=title, overwrite=False, **dict_kwargs)

        for name in cls._array_names:
            filename = folder / f'{name}.npy'
            if filename.exists():
                arr = np.load(filename)
            else:
                arr = None
            setattr(bench, name, arr)

        bench.waveforms = {}
        for key in bench.keys:
            waveforms_folder = folder / 'waveforms' / key
            if waveforms_folder.exists():
                bench.waveforms[key] = load_waveforms(waveforms_folder, with_recording=False)

        #with open(folder / 'run_times.json', 'r') as f:
        #    bench.run_times = json.load(f)

        return bench


    #def _compute_snippets_variability(self, metric='cosine'):
    #


    def compare_snippets_variability(self, metric='cosine'):
        templates = self.waveforms['static'].get_all_templates()
        
        import sklearn
        results = {}
        nb_templates = len(self.waveforms['static'].unit_ids)

        for key in self.keys:
            results[key] = np.zeros(nb_templates)
            for unit_ind, unit_id in enumerate(self.waveforms[key].sorting.unit_ids):
                w = self.waveforms[key].get_waveforms(unit_id)
                nb_waveforms = len(w)
                flat_w = w.reshape(nb_waveforms, -1)
                if metric == 'euclidean':
                    d = sklearn.metrics.pairwise_distances(templates[unit_ind].reshape(1, -1), flat_w)[0]
                elif metric == 'cosine':
                    d = sklearn.metrics.pairwise.cosine_similarity(templates[unit_ind].reshape(1, -1), flat_w)[0]
                results[key][unit_ind] = d.mean()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        colors = 0
        labels = []
        for key in self.keys:
            axes[0, 0].violinplot(results[key], [colors], showmeans=True)
            colors += 1

        _simpleaxis(axes[0, 0])
        axes[0, 0].set_xticks(np.arange(len(self.keys)), self.keys)
        if metric == 'euclidean':
            axes[0, 0].set_ylabel(r'$\| snippet - template\|_2$')
        elif metric == 'cosine':
            axes[0, 0].set_ylabel('cosine(snippet, template)')


        distances = {}
        for count, key in enumerate(['drifting', 'corrected']):
            new_templates = self.waveforms[key].get_all_templates().reshape(nb_templates, -1)
            if metric == 'euclidean':
                distances[key] = sklearn.metrics.pairwise_distances(templates.reshape(nb_templates, -1), new_templates)[0]
            elif metric == 'cosine':
                distances[key] = sklearn.metrics.pairwise.cosine_similarity(templates.reshape(nb_templates, -1), new_templates)[0]

        axes[0, 1].scatter(np.diag(distances['drifting']), np.diag(distances['corrected']), c=f'C{count}', alpha=0.5)
        xmin, xmax = axes[0, 1].get_xlim()
        axes[0, 1].plot([xmin, xmax], [xmin, xmax], 'k--')
        _simpleaxis(axes[0, 1])
        if metric == 'euclidean':
            axes[0, 1].set_xlabel(r'$\|drift - static\|_2$')
            axes[0, 1].set_ylabel(r'$\|corrected - static\|_2$')
        elif metric == 'cosine':
            axes[0, 1].set_xlabel(r'$cosine(drift, static)$')
            axes[0, 1].set_ylabel(r'$cosine(corrected, static)$')

        import MEArec as mr
        recgen = mr.load_recordings(self.mearec_filenames['static'])
        nb_templates, nb_versions, _ = recgen.template_locations.shape
        template_positions = recgen.template_locations[:, nb_versions//2, 1:3]
        distances_to_center = template_positions[:, 1]

        diff_corrected = results['corrected'] - results['static']
        diff_drifting = results['drifting'] - results['static']
        axes[1, 0].scatter(np.linalg.norm(templates, axis=(1, 2)), diff_corrected, color='C2')
        axes[1, 0].scatter(np.linalg.norm(templates, axis=(1, 2)), diff_drifting, color='C1')
        if metric == 'euclidean':
            axes[1, 0].set_ylabel(r'$\Delta \|~\|_2$')
        elif metric == 'cosine':
            axes[1, 0].set_ylabel(r'$\Delta cosine$')
        axes[1, 0].set_xlabel('template norm')
        xmin, xmax = axes[1, 0].get_xlim()
        axes[1, 0].plot([xmin, xmax], [0, 0], 'k--')
        _simpleaxis(axes[1, 0])

        axes[1, 1].scatter(distances_to_center, diff_drifting, color='C1')
        axes[1, 1].scatter(distances_to_center, diff_corrected, color='C2')
        if metric == 'euclidean':
            axes[1, 1].set_ylabel(r'$\Delta \|~\|_2$')
        elif metric == 'cosine':
            axes[1, 1].set_ylabel(r'$\Delta cosine$')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('depth (um)')
        xmin, xmax = axes[1, 1].get_xlim()
        axes[1, 1].plot([xmin, xmax], [0, 0], 'k--')
        _simpleaxis(axes[1, 1])

    def compare_residuals(self, time_range=None):

        residuals = {}

        for key in ['drifting', 'corrected']:
            difference = ResidualRecording(self.recordings['static'], self.recordings[key])
            residuals[key] = np.zeros((self.recordings['static'].get_num_channels(), 0))
            fr = int(self.recordings['static'].get_sampling_frequency())
            duration = int(self.recordings['static'].get_total_duration())
            if time_range is None:
                t_start = 0
                t_stop = duration
            else:
                t_start, t_stop = time_range

            for i in np.arange(t_start*fr, t_stop*fr, fr):
                data = np.linalg.norm(difference.get_traces(start_frame=i, end_frame=i+fr), axis=0)/np.sqrt(fr)
                residuals[key] = np.hstack((residuals[key], data[:,np.newaxis]))


        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

    #def compare_sortings(self, sorters=['spykingcircus2', 'kilosort2', 'kilosort3']):





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


def _simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()