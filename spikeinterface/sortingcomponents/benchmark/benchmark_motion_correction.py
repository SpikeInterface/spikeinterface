
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

from spikeinterface.core import extract_waveforms

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
        self.keys = ['drifting', 'static', 'corrected']      
        self.mearec_filenames['drifting'] = mearec_filename_drifting
        self.mearec_filenames['static'] = mearec_filename_static
        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins

        self.recordings = {}
        self.sortings = {}

        for key in ['drifting', 'static']:
            self.recordings[key], self.sortings[key] = read_mearec(self.mearec_filenames[key])
        
        self.title = title
        self.job_kwargs = job_kwargs
        self.correct_motion_kwargs = correct_motion_kwargs
        self.overwrite = overwrite
        self.output_folder = output_folder


    def run(self):
        self.recordings['corrected'] = CorrectMotionRecording(self.recordings['drifting'], self.motion, self.temporal_bins, self.spatial_bins)
        self.sortings['corrected'] = self.sortings['static']

        if self.output_folder is not None:
            if self.output_folder.exists() and not self.overwrite:
                raise ValueError(f"The folder {self.output_folder} is not empty")

        if self.output_folder is not None:
            self.save_to_folder(self.output_folder)

    def extract_waveforms(self):

        self.waveforms = {}
        for key in self.keys:
            if self.output_folder is None:
                mode = 'memory'
                waveforms_folder = None
            else:
                mode = 'memmap'
                waveforms_folder = self.output_folder / "waveforms" / key
            self.waveforms[key] = extract_waveforms(self.recordings[key], self.sortings[key], waveforms_folder, mode, 
                load_if_exists=True, overwrite=self.overwrite, **self.job_kwargs)

    _array_names = ('motion', 'temporal_bins', 'spatial_bins')
    _dict_kwargs_names = ('job_kwargs', 'correct_motion_kwargs', 'mearec_filenames')

    def save_to_folder(self, folder):

        if folder.exists():
            if self.overwrite:
                import shutil
                shutil.rmtree(folder)
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

        #with open(folder / 'run_times.json', 'r') as f:
        #    bench.run_times = json.load(f)

        return bench