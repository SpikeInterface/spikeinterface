
import json
import numpy as np
import time
from pathlib import Path

from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion

from spikeinterface.widgets import plot_probe_map

import scipy.interpolate

import matplotlib.pyplot as plt

import MEArec as mr

class BenchmarkMotionEstimationMearec:
    
    def __init__(self, mearec_filename, 
                detect_kwargs={},
                select_kwargs=None,
                localize_kwargs={},
                estimate_motion_kwargs={},
                output_folder=None,
                jobs_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}):
                
        self.mearec_filename = mearec_filename
        self.recording, self.gt_sorting = read_mearec(self.mearec_filename)
        
        self.job_kwargs = jobs_kwargs
        self.detect_kwargs = detect_kwargs
        self.select_kwargs = select_kwargs
        self.localize_kwargs = localize_kwargs
        self.estimate_motion_kwargs = estimate_motion_kwargs

        self.output_folder = output_folder


    def run(self):
        t0 = time.perf_counter()
        self.peaks = detect_peaks(self.recording, **self.detect_kwargs, **self.job_kwargs)
        t1 = time.perf_counter()
        if self.select_kwargs is not None:
            self.selected_peaks = select_peaks(self.peaks, **self.select_kwargs, **self.job_kwargs)
        else:
            self.selected_peaks = self.peaks
        t2 = time.perf_counter()
        self.peak_locations = localize_peaks(self.recording, self.selected_peaks, **self.localize_kwargs, **self.job_kwargs)
        t3 = time.perf_counter()
        self.motion, self.temporal_bins, self.spatial_bins = estimate_motion(self.recording, self.selected_peaks, self.peak_locations, 
                                        **self.estimate_motion_kwargs)
        t4 = time.perf_counter()

        self.run_times = dict(
            detect_peaks=t1 -t0,
            select_peaks=t2 - t1,
            localize_peaks=t3 - t2,
            estimate_motion=t4 - t3,
        )

        self.compute_gt_motion()

        ## save folder
        if self.output_folder is not None:
            self.save_to_folder(self.output_folder)


    def compute_gt_motion(self):
        self.gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filename, time_vector=self.temporal_bins)
        unit_motions = self.gt_unit_positions - self.gt_unit_positions[0, :]
        unit_positions = np.mean(self.gt_unit_positions, axis=0)

        if self.spatial_bins is None:
            self.gt_motion = np.mean(unit_motions, axis=1)[:, None]
        else:
            # time, units
            self.gt_motion = np.zeros_like(self.motion)
            for t in range(self.gt_unit_positions.shape[0]):
                f = scipy.interpolate.interp1d(unit_positions, unit_motions[t, :])
                self.gt_motion[t, :] = f(self.spatial_bins)

    def save_to_folder(self, folder):
        assert not folder.exists()
        folder.mkdir(parents=True)

        folder = Path(folder)

        for attr_name in ['gt_unit_positions', 'peaks', 'selected_peaks', 'motion', 'temporal_bins', 
                    'spatial_bins', 'peak_locations', 'gt_motion']:
            value = getattr(self, attr_name)
            if value is not None:
                np.save(folder / f'{attr_name}.npy', value)
        
        for attr_name in ('job_kwargs', 'detect_kwargs', 'select_kwargs', 'localize_kwargs', 'estimate_motion_kwargs'):
            d = getattr(self, attr_name)
            file = folder / f'{attr_name}.json'
            if d is not None:
                file.write_text(json.dumps(d, indent=4), encoding='utf8')

        run_times_filename = folder / 'run_times.json'
        run_times_filename.write_text(json.dumps(self.run_times, indent=4),encoding='utf8')

    @classmethod
    def load_from_folder(cls):
        pass



    def plot_drift(self, scaling_probe=1.5):
                
        fig, axs = plt.subplots(ncols=2)

        ax = axs[0]
        plot_probe_map(self.recording, ax=ax)
        _simpleaxis(ax)

        mr_recording = mr.load_recordings(self.mearec_filename)
            
        for loc in mr_recording.template_locations:
            if len(mr_recording.template_locations.shape) == 3:
                ax.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.7, lw=2)
            else:
                ax.scatter([loc[1]], [loc[2]], alpha=0.7, s=100)
    
        ymin, ymax = ax.get_ylim()
        ax.set_ylabel('depth (um)')
        ax.set_xlabel('depth (um)')

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)

        ax = axs[1]
        for i in range(self.gt_unit_positions.shape[1]):
            ax.plot(self.temporal_bins, self.gt_unit_positions[:, i], alpha=0.5, ls='--')
        
        if self.spatial_bins is None:
            center = (probe_y_min + probe_y_max)//2
            ax.plot(self.temporal_bins, self.gt_motion[:, 0] + center, color='red', lw=2)
        else:
            for i in range(self.gt_motion.shape[1]):
                depth = self.spatial_bins[i]
                ax.plot(self.temporal_bins, self.gt_motion[:, i] + depth, color='red', lw=2)


        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('time (s)')
        _simpleaxis(ax)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [probe_y_min, probe_y_min], 'k--', alpha=0.5)
        ax.plot([xmin, xmax], [probe_y_max, probe_y_max], 'k--', alpha=0.5)



def _simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()