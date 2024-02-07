from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np


from spikeinterface.core import load_waveforms, NpzSortingExtractor
from spikeinterface.core.core_tools import check_json


def _simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


class BenchmarkBase:
    _array_names = ()
    _waveform_names = ()
    _sorting_names = ()

    _array_names_from_parent = ()
    _waveform_names_from_parent = ()
    _sorting_names_from_parent = ()

    def __init__(
        self,
        folder=None,
        title="",
        overwrite=None,
        job_kwargs={"chunk_duration": "1s", "n_jobs": -1, "progress_bar": True, "verbose": True},
        parent_benchmark=None,
    ):
        self.folder = Path(folder)
        self.title = title
        self.overwrite = overwrite
        self.job_kwargs = job_kwargs
        self.run_times = None

        self._args = []
        self._kwargs = dict(title=title, overwrite=overwrite, job_kwargs=job_kwargs)

        self.waveforms = {}
        self.sortings = {}

        self.parent_benchmark = parent_benchmark

        if self.parent_benchmark is not None:
            for name in self._array_names_from_parent:
                setattr(self, name, getattr(parent_benchmark, name))

            for name in self._waveform_names_from_parent:
                self.waveforms[name] = parent_benchmark.waveforms[name]

            for key in parent_benchmark.sortings.keys():
                if isinstance(key, str) and key in self._sorting_names_from_parent:
                    self.sortings[key] = parent_benchmark.sortings[key]
                elif isinstance(key, tuple) and key[0] in self._sorting_names_from_parent:
                    self.sortings[key] = parent_benchmark.sortings[key]

    def save_to_folder(self):
        if self.folder.exists():
            import glob, os

            pattern = "*.*"
            files = self.folder.glob(pattern)
            for file in files:
                if file.is_file():
                    os.remove(file)
        else:
            self.folder.mkdir(parents=True)

        if self.parent_benchmark is None:
            parent_folder = None
        else:
            parent_folder = str(self.parent_benchmark.folder)

        info = {
            "args": self._args,
            "kwargs": self._kwargs,
            "parent_folder": parent_folder,
        }
        info = check_json(info)
        (self.folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        for name in self._array_names:
            if self.parent_benchmark is not None and name in self._array_names_from_parent:
                continue
            value = getattr(self, name)
            if value is not None:
                np.save(self.folder / f"{name}.npy", value)

        if self.run_times is not None:
            run_times_filename = self.folder / "run_times.json"
            run_times_filename.write_text(json.dumps(self.run_times, indent=4), encoding="utf8")

        for key, sorting in self.sortings.items():
            (self.folder / "sortings").mkdir(exist_ok=True)
            if isinstance(key, str):
                npz_file = self.folder / "sortings" / (str(key) + ".npz")
            elif isinstance(key, tuple):
                npz_file = self.folder / "sortings" / ("_###_".join(key) + ".npz")
            NpzSortingExtractor.write_sorting(sorting, npz_file)

    @classmethod
    def load_from_folder(cls, folder, parent_benchmark=None):
        folder = Path(folder)
        assert folder.exists()

        with open(folder / "info.json", "r") as f:
            info = json.load(f)
        args = info["args"]
        kwargs = info["kwargs"]

        if info["parent_folder"] is None:
            parent_benchmark = None
        else:
            if parent_benchmark is None:
                parent_benchmark = cls.load_from_folder(info["parent_folder"])

        import os

        kwargs["folder"] = folder

        bench = cls(*args, **kwargs, parent_benchmark=parent_benchmark)

        for name in cls._array_names:
            filename = folder / f"{name}.npy"
            if filename.exists():
                arr = np.load(filename)
            else:
                arr = None
            setattr(bench, name, arr)

        if (folder / "run_times.json").exists():
            with open(folder / "run_times.json", "r") as f:
                bench.run_times = json.load(f)
        else:
            bench.run_times = None

        for key in bench._waveform_names:
            if parent_benchmark is not None and key in bench._waveform_names_from_parent:
                continue
            waveforms_folder = folder / "waveforms" / key
            if waveforms_folder.exists():
                bench.waveforms[key] = load_waveforms(waveforms_folder, with_recording=True)

        sorting_folder = folder / "sortings"
        if sorting_folder.exists():
            for npz_file in sorting_folder.glob("*.npz"):
                name = npz_file.stem
                if "_###_" in name:
                    key = tuple(name.split("_###_"))
                else:
                    key = name
                bench.sortings[key] = NpzSortingExtractor(npz_file)

        return bench
