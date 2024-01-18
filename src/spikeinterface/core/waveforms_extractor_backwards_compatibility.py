"""
This backwards compatibility module aims to:
  * load old WaveformsExtractor saved with folder or zarr  (version <=0.100) into the  SortingResult (version>0.100)
  * mock the function extract_waveforms() and the class SortingResult() but based SortingResult
"""
from __future__ import annotations

from typing import Literal, Optional

from pathlib import Path

import numpy as np


from .baserecording import BaseRecording
from .basesorting import BaseSorting
from .sortingresult import start_sorting_result
from .job_tools import split_job_kwargs


_backwards_compatibility_msg = """####
# extract_waveforms() and WaveformExtractor() have been replace by SortingResult since version 0.101
# You should use start_sorting_result() instead.
# extract_waveforms() is now mocking the old behavior for backwards compatibility only and will be removed after 0.103
####"""


def extract_waveforms(
    recording,
    sorting,
    folder=None,
    mode="folder",
    precompute_template=("average",),
    ms_before=1.0,
    ms_after=2.0,
    max_spikes_per_unit=500,
    overwrite=None,
    return_scaled=True,
    dtype=None,
    sparse=True,
    sparsity=None,
    sparsity_temp_folder=None,
    num_spikes_for_sparsity=100,
    unit_batch_size=None,
    allow_unfiltered=None,
    use_relative_path=None,
    seed=None,
    load_if_exists=None,
    **kwargs,
):
    """
    This mock the extract_waveforms() in version <= 0.100 to not break old codes but using
    the SortingResult (version >0.100) internally.

    This return a MockWaveformExtractor object that mock the old WaveformExtractor
    """
    print(_backwards_compatibility_msg)

    assert load_if_exists is None, "load_if_exists=True/False is not supported anymore. use load_if_exists=None"
    assert overwrite is None, "overwrite=True/False is not supported anymore. use overwrite=None"

    other_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if mode == "folder":
        assert folder is not None
        folder = Path(folder)
        format = "binary_folder"
    else:
        mode = "memory"

    assert sparsity_temp_folder is None, "sparsity_temp_folder must be None"
    assert unit_batch_size is None, "unit_batch_size must be None"

    if use_relative_path is not None:
        print("use_relative_path is ignored")

    if allow_unfiltered is not None:
        print("allow_unfiltered is ignored")

    sparsity_kwargs = dict(
        num_spikes_for_sparsity=num_spikes_for_sparsity,
        ms_before=ms_before,
        ms_after=ms_after,
        **other_kwargs,
        **job_kwargs
    )
    sorting_result = start_sorting_result(sorting, recording, format=format, folder=folder, 
        sparse=sparse, sparsity=sparsity, **sparsity_kwargs
    )

    # TODO propagate job_kwargs

    sorting_result.select_random_spikes(max_spikes_per_unit=max_spikes_per_unit, seed=seed)

    waveforms_params = dict(ms_before=ms_before, ms_after=ms_after, return_scaled=return_scaled, dtype=dtype)
    sorting_result.compute("waveforms", **waveforms_params)

    templates_params = dict(operators=list(precompute_template))
    sorting_result.compute("templates", **templates_params)

    we = MockWaveformExtractor(sorting_result)

    return we



class MockWaveformExtractor:
    def __init__(self, sorting_result):
        self.sorting_result = sorting_result

    def __repr__(self):
        txt = "MockWaveformExtractor: mock the old WaveformExtractor with "
        txt += self.sorting_result.__repr__()
        return txt

    def is_sparse(self) -> bool:
        return self.sorting_result.is_sparse()
    
    def has_waveforms(self) -> bool:

        raise NotImplementedError

    def delete_waveforms(self) -> None:
        raise NotImplementedError

    @property
    def recording(self) -> BaseRecording:
        raise NotImplementedError

    @property
    def channel_ids(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def sampling_frequency(self) -> float:
        raise NotImplementedError

    @property
    def unit_ids(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def nbefore(self) -> int:
        raise NotImplementedError

    @property
    def nafter(self) -> int:
        raise NotImplementedError

    @property
    def nsamples(self) -> int:
        return self.nbefore + self.nafter

    @property
    def return_scaled(self) -> bool:
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    def is_read_only(self) -> bool:
        raise NotImplementedError

    def has_recording(self) -> bool:
        raise NotImplementedError

    def get_num_samples(self, segment_index: Optional[int] = None) -> int:
        raise NotImplementedError

    def get_total_samples(self) -> int:
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_samples(segment_index)
        return s

    def get_total_duration(self) -> float:
        duration = self.get_total_samples() / self.sampling_frequency
        return duration

    def get_num_channels(self) -> int:
        raise NotImplementedError
        # if self.has_recording():
        #     return self.recording.get_num_channels()
        # else:
        #     return self._rec_attributes["num_channels"]

    def get_num_segments(self) -> int:
        return self.sorting_result.sorting.get_num_segments()

    def get_probegroup(self):
        raise NotImplementedError
        # if self.has_recording():
        #     return self.recording.get_probegroup()
        # else:
        #     return self._rec_attributes["probegroup"]

    # def is_filtered(self) -> bool:
    #     if self.has_recording():
    #         return self.recording.is_filtered()
    #     else:
    #         return self._rec_attributes["is_filtered"]

    def get_probe(self):
        probegroup = self.get_probegroup()
        assert len(probegroup.probes) == 1, "There are several probes. Use `get_probegroup()`"
        return probegroup.probes[0]

    def get_channel_locations(self) -> np.ndarray:
        raise NotImplementedError

    def channel_ids_to_indices(self, channel_ids) -> np.ndarray:
        raise NotImplementedError

    def get_recording_property(self, key) -> np.ndarray:
        raise NotImplementedError

    def get_sorting_property(self, key) -> np.ndarray:
        return self.sorting.get_property(key)

    # def has_extension(self, extension_name: str) -> bool:
    #     raise NotImplementedError
    
    def get_waveforms(
        self,
        unit_id,
        with_index: bool = False,
        cache: bool = False,
        lazy: bool = True,
        sparsity=None,
        force_dense: bool = False,
    ):
        raise NotImplementedError
        
    def get_sampled_indices(self, unit_id):
        raise NotImplementedError


    def get_all_templates(
        self, unit_ids: list | np.array | tuple | None = None, mode="average", percentile: float | None = None
    ):
        raise NotImplementedError

    def get_template(
        self, unit_id, mode="average", sparsity=None, force_dense: bool = False, percentile: float | None = None
    ):
        raise NotImplementedError



def load_waveforms(folder, with_recording: bool = True, sorting: Optional[BaseSorting] = None, output="SortingResult"):
    """
    This read an old WaveformsExtactor folder (folder or zarr) and convert it into a SortingResult or MockWaveformExtractor.

    """

    raise NotImplementedError

    # This will be something like this create a SortingResult in memory and copy/translate all data into the new structure.
    # sorting_result = ...

    # if output == "SortingResult":
    #     return sorting_result
    # elif output in ("WaveformExtractor", "MockWaveformExtractor"):
    #     return MockWaveformExtractor(sorting_result)
