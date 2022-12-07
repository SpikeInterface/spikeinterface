from pathlib import Path
import shutil
import json
import numpy as np


from spikeinterface.core import load_waveforms


def _simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

class BenchmarkBase:
    _array_names = ()
    _waveform_names = ()
    _array_names_from_other = None

    def __init__(self, folder=None, title='', overwrite=None, 
                job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}):
        self.folder = Path(folder)
        self.title = title
        self.overwrite = overwrite
        self.job_kwargs = job_kwargs
        self.run_times = None

        self._args = []
        self._kwargs = dict(title=title, overwrite=overwrite, job_kwargs=job_kwargs)

        self.waveforms = {}


    def save_to_folder(self):
        if self.folder.exists():
            if self.overwrite:
                shutil.rmtree(self.folder)

        self.folder.mkdir(parents=True)

        info = {
            'args': self._args,
            'kwargs': self._kwargs,
        }
        (self.folder / 'info.json').write_text(json.dumps(info, indent=4), encoding='utf8')

        for name in self._array_names:
            value = getattr(self, name)
            if value is not None:
                np.save(self.folder / f'{name}.npy', value)
        
        for name in self._kwargs:
            d = getattr(self, name)
            file = self.folder / f'{name}.json'
            if d is not None:
                file.write_text(json.dumps(d, indent=4), encoding='utf8')

        if self.run_times is not None:
            run_times_filename = self.folder / 'run_times.json'
            run_times_filename.write_text(json.dumps(self.run_times, indent=4),encoding='utf8')

    @classmethod
    def load_from_folder(cls, folder):
        folder = Path(folder)
        assert folder.exists()

        with open(folder / 'info.json', 'r') as f:
            info = json.load(f)
        args = info['args']
        kwargs = info['kwargs']

        import os
        kwargs['folder'] = str(os.path.abspath(folder))
        bench = cls(*args, **kwargs)

        # dict_kwargs = dict()
        # for name in cls._dict_kwargs_names:
        #     filename = folder / f'{name}.json' 
        #     if filename.exists():
        #         with open(filename, 'r') as f:
        #             d = json.load(f)
        #     else:
        #         d = None
        #     dict_kwargs[name] = d

        for name in cls._array_names:
            filename = folder / f'{name}.npy'
            if filename.exists():
                arr = np.load(filename)
            else:
                arr = None
            setattr(bench, name, arr)

        if (folder / 'run_times.json').exists():
            with open(folder / 'run_times.json', 'r') as f:
                bench.run_times = json.load(f)
        else:
            bench.run_times = None


        for key in bench._waveform_names:
            waveforms_folder = folder / 'waveforms' / key
            if waveforms_folder.exists():
                bench.waveforms[key] = load_waveforms(waveforms_folder, with_recording=False)

        return bench
