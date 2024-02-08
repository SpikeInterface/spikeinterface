"""
This backwards compatibility module aims to:
  * load old WaveformsExtractor saved with folder or zarr  (version <=0.100) into the  SortingResult (version>0.100)
  * mock the function extract_waveforms() and the class SortingResult() but based SortingResult
"""
from __future__ import annotations

from typing import Literal, Optional

from pathlib import Path

import json

import numpy as np

import probeinterface

from .baserecording import BaseRecording
from .basesorting import BaseSorting
from .sortingresult import start_sorting_result
from .job_tools import split_job_kwargs
from .sparsity import ChannelSparsity
from .sortingresult import SortingResult, load_sorting_result
from .base import load_extractor
from .result_core import ComputeWaveforms, ComputeTemplates

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
        folder = None
        format = "memory"

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
    sorting_result.compute("waveforms", **waveforms_params, **job_kwargs)

    templates_params = dict(operators=list(precompute_template))
    sorting_result.compute("templates", **templates_params)

    # this also done because some metrics need it
    sorting_result.compute("noise_levels")
    

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
        return self.sorting_result.get_extension("waveforms") is not None

    def delete_waveforms(self) -> None:
        self.sorting_result.delete_extension("waveforms")

    @property
    def recording(self) -> BaseRecording:
        return self.sorting_result.recording
    
    @property
    def sorting(self) -> BaseSorting:
        return self.sorting_result.sorting

    @property
    def channel_ids(self) -> np.ndarray:
        return self.sorting_result.channel_ids

    @property
    def sampling_frequency(self) -> float:
        return self.sorting_result.sampling_frequency

    @property
    def unit_ids(self) -> np.ndarray:
        return self.sorting_result.unit_ids

    @property
    def nbefore(self) -> int:
        ms_before = self.sorting_result.get_extension("waveforms").params["ms_before"]
        return int(ms_before * self.sampling_frequency / 1000.0)

    @property
    def nafter(self) -> int:
        ms_after = self.sorting_result.get_extension("waveforms").params["ms_after"]
        return int(ms_after * self.sampling_frequency / 1000.0)

    @property
    def nsamples(self) -> int:
        return self.nbefore + self.nafter

    @property
    def return_scaled(self) -> bool:
        return self.sorting_result.get_extension("waveforms").params["return_scaled"]

    @property
    def dtype(self):
        return self.sorting_result.get_extension("waveforms").params["dtype"]

    def is_read_only(self) -> bool:
        return self.sorting_result.is_read_only()

    def has_recording(self) -> bool:
        return self.sorting_result._recording is not None

    def get_num_samples(self, segment_index: Optional[int] = None) -> int:
        return self.sorting_result.get_num_samples(segment_index)

    def get_total_samples(self) -> int:
        return self.sorting_result.get_total_samples()

    def get_total_duration(self) -> float:
        return self.sorting_result.get_total_duration()

    def get_num_channels(self) -> int:
        return self.sorting_result.get_num_channels()

    def get_num_segments(self) -> int:
        return self.sorting_result.get_num_segments()

    def get_probegroup(self):
        return self.sorting_result.get_probegroup()

    def get_probe(self):
        return self.sorting_result.get_probe()

    def is_filtered(self) -> bool:
        return self.sorting_result.rec_attributes["is_filtered"]

    def get_channel_locations(self) -> np.ndarray:
        return self.sorting_result.get_channel_locations()

    def channel_ids_to_indices(self, channel_ids) -> np.ndarray:
        return self.sorting_result.channel_ids_to_indices(channel_ids)

    def get_recording_property(self, key) -> np.ndarray:
        return self.sorting_result.get_recording_property(key)

    def get_sorting_property(self, key) -> np.ndarray:
        return self.sorting_result.get_sorting_property(key)

    @property
    def sparsity(self):
        return self.sorting_result.sparsity

    @property
    def folder(self):
        if self.sorting_result.format != "memory":
            return self.sorting_result.folder

    def has_extension(self, extension_name: str) -> bool:
        return self.sorting_result.has_extension(extension_name)

    def get_sampled_indices(self, unit_id):
        # In Waveforms extractor "selected_spikes" was a dict (key: unit_id) with a complex dtype as follow
        selected_spikes = []
        for segment_index in range(self.get_num_segments()):
            inds = self.sorting_result.get_selected_indices_in_spike_train(unit_id, segment_index)
            sampled_index = np.zeros(inds.size, dtype=[("spike_index", "int64"), ("segment_index", "int64")])
            sampled_index["spike_index"] = inds
            sampled_index["segment_index"][:] = segment_index
            selected_spikes.append(sampled_index)
        return np.concatenate(selected_spikes)

    def get_waveforms(
        self,
        unit_id,
        with_index: bool = False,
        cache: bool = False,
        lazy: bool = True,
        sparsity=None,
        force_dense: bool = False,
    ):  
        # lazy and cache are ingnored
        ext = self.sorting_result.get_extension("waveforms")
        unit_index = self.sorting.id_to_index(unit_id)
        spikes = self.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        spike_mask = some_spikes["unit_index"] == unit_index
        wfs = ext.data["waveforms"][spike_mask, :, :]

        if sparsity is not None:
            assert self.sorting_result.sparsity is None, "Waveforms are alreayd sparse! Cannot apply an additional sparsity."
            wfs = wfs[:, :, sparsity.mask[self.sorting.id_to_index(unit_id)]]

        if force_dense:
            assert sparsity is None
            if self.sorting_result.sparsity is None:
                # nothing to do
                pass
            else:
                num_channels = self.get_num_channels()
                dense_wfs = np.zeros((wfs.shape[0], wfs.shape[1], num_channels), dtype=np.float32)
                unit_sparsity = self.sorting_result.sparsity.mask[unit_index]
                dense_wfs[:, :, unit_sparsity] = wfs
                wfs = dense_wfs

        if with_index:
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def get_all_templates(
        self, unit_ids: list | np.array | tuple | None = None, mode="average", percentile: float | None = None
    ):
        ext = self.sorting_result.get_extension("templates")

        if mode == "percentile":
            key = f"pencentile_{percentile}"
        else:
            key = mode
        
        templates = ext.data.get(key)
        if templates is None:
            raise ValueError(f"{mode} is not computed")

        if unit_ids is not None:
            unit_indices = self.sorting.ids_to_indices(unit_ids)
            templates = templates[unit_indices, :, :]

        return templates


    def get_template(
        self, unit_id, mode="average", sparsity=None, force_dense: bool = False, percentile: float | None = None
    ):
        # force_dense and sparsity are ignored
        templates = self.get_all_templates(unit_ids=[unit_id], mode=mode, percentile=percentile)
        return templates[0]



def load_waveforms(folder, with_recording: bool = True, sorting: Optional[BaseSorting] = None, output="SortingResult", ):
    """
    This read an old WaveformsExtactor folder (folder or zarr) and convert it into a SortingResult or MockWaveformExtractor.

    It also mimic the old load_waveforms by opening a Sortingresult folder and return a MockWaveformExtractor.

    """

    folder = Path(folder)
    assert folder.is_dir(), "Waveform folder does not exists"

    if (folder / "spikeinterface_info.json").exists:
        with open(folder / "spikeinterface_info.json", mode="r") as f:
            info = json.load(f)
        if info.get("object", None) == "SortingResult":
            # in this case the folder is already a sorting result from version >= 0.101.0 but create with the MockWaveformExtractor
            sorting_result = load_sorting_result(folder)
            sorting_result.load_all_saved_extension()
            we = MockWaveformExtractor(sorting_result)
            return we

    if folder.suffix == ".zarr":
        raise NotImplementedError
        # Alessio this is for you
    else:
        sorting_result = _read_old_waveforms_extractor_binary(folder)

    if output == "SortingResult":
        return sorting_result
    elif output in ("WaveformExtractor", "MockWaveformExtractor"):
        return MockWaveformExtractor(sorting_result)



def _read_old_waveforms_extractor_binary(folder):
    params_file = folder / "params.json"
    if not params_file.exists():
        raise ValueError(f"This folder is not a WaveformsExtractor folder {folder}")
    with open(params_file, "r") as f:
        params = json.load(f)

    sparsity_file = folder / "sparsity.json"
    if params_file.exists():
        with open(sparsity_file, "r") as f:
            sparsity_dict = json.load(f)
            sparsity = ChannelSparsity.from_dict(sparsity_dict)
    else:
        sparsity = None

    # recording attributes
    rec_attributes_file = folder / "recording_info" / "recording_attributes.json"
    with open(rec_attributes_file, "r") as f:
        rec_attributes = json.load(f)
    probegroup_file = folder / "recording_info" / "probegroup.json"
    if probegroup_file.is_file():
        rec_attributes["probegroup"] = probeinterface.read_probeinterface(probegroup_file)
    else:
        rec_attributes["probegroup"] = None

    # recording
    recording = None
    if (folder / "recording.json").exists():
        try:
            recording = load_extractor(folder / "recording.json", base_folder=folder)
        except:
            pass
    elif (folder / "recording.pickle").exists():
        try:
            recording = load_extractor(folder / "recording.pickle", base_folder=folder)
        except:
            pass

    # sorting
    if (folder / "sorting.json").exists():
        sorting = load_extractor(folder / "sorting.json", base_folder=folder)
    elif (folder / "sorting.pickle").exists():
        sorting = load_extractor(folder / "sorting.pickle", base_folder=folder)

    sorting_result = SortingResult.create_memory(sorting, recording, sparsity, rec_attributes=rec_attributes)

    # waveforms
    # need to concatenate all waveforms in one unique buffer
    # need to concatenate sampled_index and order it
    waveform_folder = folder / "waveforms"
    if waveform_folder.exists():
        
        spikes = sorting.to_spike_vector()
        random_spike_mask = np.zeros(spikes.size, dtype="bool")

        all_sampled_indices = []
        # first readd all sampled_index to get the correct ordering
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            # unit_indices has dtype=[("spike_index", "int64"), ("segment_index", "int64")] 
            unit_indices = np.load(waveform_folder / f"sampled_index_{unit_id}.npy")
            for segment_index in range(sorting.get_num_segments()):
                in_seg_selected = unit_indices[unit_indices["segment_index"] == segment_index]["spike_index"]
                spikes_indices = np.flatnonzero((spikes["unit_index"] == unit_index) & (spikes["segment_index"] == segment_index))
                random_spike_mask[spikes_indices[in_seg_selected]] = True
        random_spikes_indices = np.flatnonzero(random_spike_mask)

        num_spikes = random_spikes_indices.size
        if sparsity is None:
            max_num_channel = len(rec_attributes["channel_ids"])
        else:
            max_num_channel = np.max(np.sum(sparsity.mask, axis=1))

        nbefore = int(params["ms_before"] * sorting.sampling_frequency / 1000.0)
        nafter = int(params["ms_after"] * sorting.sampling_frequency / 1000.0)

        waveforms = np.zeros((num_spikes, nbefore + nafter, max_num_channel), dtype=params["dtype"])
        # then read waveforms per units
        some_spikes = spikes[random_spikes_indices]
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            wfs = np.load(waveform_folder / f"waveforms_{unit_id}.npy")
            mask = some_spikes["unit_index"] == unit_index
            waveforms[:, :, :wfs.shape[2]][mask, :, :] = wfs

        sorting_result.random_spikes_indices = random_spikes_indices

        ext = ComputeWaveforms(sorting_result)
        ext.params = params
        ext.data["waveforms"] = waveforms
        sorting_result.extensions["waveforms"] = ext

    # templates saved dense
    # load cached templates
    templates = {}
    for mode in ("average", "std", "median", "percentile"):
        template_file = folder / f"templates_{mode}.npy"
        if template_file.is_file():
            templates [mode] = np.load(template_file)
    if len(templates) > 0:
        ext = ComputeTemplates(sorting_result)
        ext.params = dict(operators=list(templates.keys()))
        for mode, arr in templates.items():
            ext.data[mode] = arr
        sorting_result.extensions["templates"] = ext


    # TODO : implement this when extension will be prted in the new API
    # old_extension_to_new_class : {
        # old extensions with same names and equvalent data
        # "spike_amplitudes": ,
        # "spike_locations": ,
        # "amplitude_scalings": ,
        # "template_metrics" : ,
        # "similarity": , 
        # "unit_locations": ,
        # "correlograms" : ,
        # isi_histograms: ,
        # "noise_levels": ,
        # "quality_metrics": ,
        # "principal_components" : , 
    # }
    # for ext_name, new_class in old_extension_to_new_class.items():
    #     ext_folder = folder / ext_name
    #     ext = new_class(sorting_result)
    #     with open(ext_folder / "params.json", "r") as f:
    #         params = json.load(f)
    #     ext.params = params
    #     if ext_name == "spike_amplitudes":
    #         amplitudes = []
    #         for segment_index in range(sorting.get_num_segments()):
    #             amplitudes.append(np.load(ext_folder / f"amplitude_segment_{segment_index}.npy"))
    #         amplitudes = np.concatenate(amplitudes)
    #         ext.data["amplitudes"] = amplitudes
    #     elif ext_name == "spike_locations":
    #         ext.data["spike_locations"] = np.load(ext_folder / "spike_locations.npy")
    #     elif ext_name == "amplitude_scalings":
    #         ext.data["amplitude_scalings"] = np.load(ext_folder / "amplitude_scalings.npy")
    #     elif ext_name == "template_metrics":
    #         import pandas as pd
    #         ext.data["metrics"] = pd.read_csv(ext_folder / "metrics.csv", index_col=0)
    #     elif ext_name == "similarity":
    #         ext.data["similarity"] = np.load(ext_folder / "similarity.npy")
    #     elif ext_name == "unit_locations":
    #         ext.data["unit_locations"] = np.load(ext_folder / "unit_locations.npy")
    #     elif ext_name == "correlograms":
    #         ext.data["ccgs"] = np.load(ext_folder / "ccgs.npy")
    #         ext.data["bins"] = np.load(ext_folder / "bins.npy")
    #     elif ext_name == "isi_histograms":
    #         ext.data["isi_histograms"] = np.load(ext_folder / "isi_histograms.npy")
    #         ext.data["bins"] = np.load(ext_folder / "bins.npy")
    #     elif ext_name == "noise_levels":
    #         ext.data["noise_levels"] = np.load(ext_folder / "noise_levels.npy")
    #     elif ext_name == "quality_metrics":
    #         import pandas as pd
    #         ext.data["metrics"] = pd.read_csv(ext_folder / "metrics.csv", index_col=0)
    #     elif ext_name == "principal_components":
    #         # TODO: this is for you
    #         pass



    return sorting_result


