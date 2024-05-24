"""
This backwards compatibility module aims to:
  * load old WaveformsExtractor saved with folder or zarr  (version <=0.100) into the  SortingAnalyzer (version>0.100)
  * mock the function extract_waveforms() and the class SortingAnalyzer() but based SortingAnalyzer
"""

from __future__ import annotations

from typing import Literal, Optional

from pathlib import Path

import json

import numpy as np

import probeinterface

from .baserecording import BaseRecording
from .basesorting import BaseSorting
from .sortinganalyzer import create_sorting_analyzer, get_extension_class
from .job_tools import split_job_kwargs
from .sparsity import ChannelSparsity
from .sortinganalyzer import SortingAnalyzer, load_sorting_analyzer
from .base import load_extractor
from .analyzer_extension_core import ComputeRandomSpikes, ComputeWaveforms, ComputeTemplates

_backwards_compatibility_msg = """####
# extract_waveforms() and WaveformExtractor() have been replaced by the `SortingAnalyzer` since version 0.101.0.
# You should use `spikeinterface.create_sorting_analyzer()` instead.
# `spikeinterface.extract_waveforms()` is now mocking the old behavior for backwards compatibility only,
# and will be removed with version 0.103.0
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
    the SortingAnalyzer (version >0.100) internally.

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
        **job_kwargs,
    )
    sorting_analyzer = create_sorting_analyzer(
        sorting,
        recording,
        format=format,
        folder=folder,
        sparse=sparse,
        sparsity=sparsity,
        return_scaled=return_scaled,
        **sparsity_kwargs,
    )

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=max_spikes_per_unit, seed=seed)

    waveforms_params = dict(ms_before=ms_before, ms_after=ms_after, dtype=dtype)
    sorting_analyzer.compute("waveforms", **waveforms_params, **job_kwargs)

    templates_params = dict(operators=list(precompute_template))
    sorting_analyzer.compute("templates", **templates_params)

    # this also done because some metrics need it
    sorting_analyzer.compute("noise_levels")

    we = MockWaveformExtractor(sorting_analyzer)

    return we


class MockWaveformExtractor:
    def __init__(self, sorting_analyzer):
        self.sorting_analyzer = sorting_analyzer

    def __repr__(self):
        txt = "MockWaveformExtractor: mock the old WaveformExtractor with "
        txt += self.sorting_analyzer.__repr__()
        return txt

    def is_sparse(self) -> bool:
        return self.sorting_analyzer.is_sparse()

    def has_waveforms(self) -> bool:
        return self.sorting_analyzer.get_extension("waveforms") is not None

    def delete_waveforms(self) -> None:
        self.sorting_analyzer.delete_extension("waveforms")

    def delete_extension(self, extension) -> None:
        self.sorting_analyzer.delete_extension()

    @property
    def recording(self) -> BaseRecording:
        return self.sorting_analyzer.recording

    @property
    def sorting(self) -> BaseSorting:
        return self.sorting_analyzer.sorting

    @property
    def channel_ids(self) -> np.ndarray:
        return self.sorting_analyzer.channel_ids

    @property
    def sampling_frequency(self) -> float:
        return self.sorting_analyzer.sampling_frequency

    @property
    def unit_ids(self) -> np.ndarray:
        return self.sorting_analyzer.unit_ids

    @property
    def nbefore(self) -> int:
        ms_before = self.sorting_analyzer.get_extension("waveforms").params["ms_before"]
        return int(ms_before * self.sampling_frequency / 1000.0)

    @property
    def nafter(self) -> int:
        ms_after = self.sorting_analyzer.get_extension("waveforms").params["ms_after"]
        return int(ms_after * self.sampling_frequency / 1000.0)

    @property
    def nsamples(self) -> int:
        return self.nbefore + self.nafter

    @property
    def return_scaled(self) -> bool:
        return self.sorting_analyzer.get_extension("waveforms").params["return_scaled"]

    @property
    def dtype(self):
        return self.sorting_analyzer.get_extension("waveforms").params["dtype"]

    def is_read_only(self) -> bool:
        return self.sorting_analyzer.is_read_only()

    def has_recording(self) -> bool:
        return self.sorting_analyzer._recording is not None

    def get_num_samples(self, segment_index: Optional[int] = None) -> int:
        return self.sorting_analyzer.get_num_samples(segment_index)

    def get_total_samples(self) -> int:
        return self.sorting_analyzer.get_total_samples()

    def get_total_duration(self) -> float:
        return self.sorting_analyzer.get_total_duration()

    def get_num_channels(self) -> int:
        return self.sorting_analyzer.get_num_channels()

    def get_num_segments(self) -> int:
        return self.sorting_analyzer.get_num_segments()

    def get_probegroup(self):
        return self.sorting_analyzer.get_probegroup()

    def get_probe(self):
        return self.sorting_analyzer.get_probe()

    def is_filtered(self) -> bool:
        return self.sorting_analyzer.rec_attributes["is_filtered"]

    def get_channel_locations(self) -> np.ndarray:
        return self.sorting_analyzer.get_channel_locations()

    def channel_ids_to_indices(self, channel_ids) -> np.ndarray:
        return self.sorting_analyzer.channel_ids_to_indices(channel_ids)

    def get_recording_property(self, key) -> np.ndarray:
        return self.sorting_analyzer.get_recording_property(key)

    def get_sorting_property(self, key) -> np.ndarray:
        return self.sorting_analyzer.get_sorting_property(key)

    def get_available_extension_names(self):
        return self.sorting_analyzer.get_loaded_extension_names()

    @property
    def sparsity(self):
        return self.sorting_analyzer.sparsity

    @property
    def folder(self):
        if self.sorting_analyzer.format != "memory":
            return self.sorting_analyzer.folder

    @property
    def format(self):
        if self.sorting_analyzer.format == "binary_folder":
            return "binary"
        else:
            return self.sorting_analyzer.format

    def has_extension(self, extension_name: str) -> bool:
        return self.sorting_analyzer.has_extension(extension_name)

    def select_units(self, unit_ids):
        return self.sorting_analyzer.select_units(unit_ids)

    def get_sampled_indices(self, unit_id):
        # In Waveforms extractor "selected_spikes" was a dict (key: unit_id) with a complex dtype as follow
        selected_spikes = []
        for segment_index in range(self.get_num_segments()):
            # inds = self.sorting_analyzer.get_selected_indices_in_spike_train(unit_id, segment_index)
            assert self.sorting_analyzer.has_extension(
                "random_spikes"
            ), "get_sampled_indices() requires the 'random_spikes' extension."
            inds = self.sorting_analyzer.get_extension("random_spikes").get_selected_indices_in_spike_train(
                unit_id, segment_index
            )

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
        ext = self.sorting_analyzer.get_extension("waveforms")
        unit_index = self.sorting.id_to_index(unit_id)

        assert self.sorting_analyzer.has_extension(
            "random_spikes"
        ), "get_sampled_indices() requires the 'random_spikes' extension."

        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        spike_mask = some_spikes["unit_index"] == unit_index
        wfs = ext.data["waveforms"][spike_mask, :, :]

        if sparsity is not None:
            assert (
                self.sorting_analyzer.sparsity is None
            ), "Waveforms are alreayd sparse! Cannot apply an additional sparsity."
            wfs = wfs[:, :, sparsity.mask[self.sorting.id_to_index(unit_id)]]

        if force_dense:
            assert sparsity is None
            if self.sorting_analyzer.sparsity is None:
                # nothing to do
                pass
            else:
                num_channels = self.get_num_channels()
                dense_wfs = np.zeros((wfs.shape[0], wfs.shape[1], num_channels), dtype=np.float32)
                unit_sparsity = self.sorting_analyzer.sparsity.mask[unit_index]
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
        ext = self.sorting_analyzer.get_extension("templates")

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


def load_waveforms(
    folder,
    with_recording: bool = True,
    sorting: Optional[BaseSorting] = None,
    output="MockWaveformExtractor",
):
    """
    This read an old WaveformsExtactor folder (folder or zarr) and convert it into a SortingAnalyzer or MockWaveformExtractor.

    It also mimic the old load_waveforms by opening a Sortingresult folder and return a MockWaveformExtractor.
    This later behavior is usefull to no break old code like this in versio >=0.101

    >>> # In this example we is a MockWaveformExtractor that behave the same as before
    >>> we = extract_waveforms(..., folder="/my_we")
    >>> we = load_waveforms("/my_we")
    >>> templates = we.get_all_templates()

    Parameters
    ----------
    folder: str | Path
        The folder to the waveform extractor (binary or zarr)
    with_recording: bool
        For back-compatibility, ignored
    sorting: BaseSorting | None, default: None
        The sorting object to instantiate with the Waveforms
    output: "MockWaveformExtractor" | "SortingAnalyzer", default: "MockWaveformExtractor"
        The output format

    Returns
    -------
    waveforms_or_analyzer: MockWaveformExtractor | SortingAnalyzer
        The returned MockWaveformExtractor or SortingAnalyzer
    """

    folder = Path(folder)
    assert folder.is_dir(), "Waveform folder does not exists"

    if (folder / "spikeinterface_info.json").exists():
        with open(folder / "spikeinterface_info.json", mode="r") as f:
            info = json.load(f)
        if info.get("object", None) == "SortingAnalyzer":
            # in this case the folder is already a sorting result from version >= 0.101.0 but create with the MockWaveformExtractor
            sorting_analyzer = load_sorting_analyzer(folder)
            sorting_analyzer.load_all_saved_extension()
            we = MockWaveformExtractor(sorting_analyzer)
            return we

    if folder.suffix == ".zarr":
        raise NotImplementedError("Waveform extractors back-compatibility from Zarr format is not supported")
        # sorting_analyzer = _read_old_waveforms_extractor_zarr(folder, sorting)
    else:
        sorting_analyzer = _read_old_waveforms_extractor_binary(folder, sorting)

    if output == "SortingAnalyzer":
        return sorting_analyzer
    elif output in ("WaveformExtractor", "MockWaveformExtractor"):
        return MockWaveformExtractor(sorting_analyzer)


# old extensions with same names and equvalent data except similarity>template_similarity
old_extension_to_new_class_map = {
    "spike_amplitudes": "spike_amplitudes",
    "spike_locations": "spike_locations",
    "amplitude_scalings": "amplitude_scalings",
    "template_metrics": "template_metrics",
    "similarity": "template_similarity",
    "unit_locations": "unit_locations",
    "correlograms": "correlograms",
    "isi_histograms": "isi_histograms",
    "noise_levels": "noise_levels",
    "quality_metrics": "quality_metrics",
    "principal_components": "principal_components",
}


def _read_old_waveforms_extractor_binary(folder, sorting):
    folder = Path(folder)
    params_file = folder / "params.json"
    if not params_file.exists():
        raise ValueError(f"This folder is not a WaveformsExtractor folder {folder}")
    with open(params_file, "r") as f:
        params = json.load(f)

    return_scaled = params["return_scaled"]

    sparsity_file = folder / "sparsity.json"
    if sparsity_file.exists():
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
    if sorting is None:
        if (folder / "sorting.json").exists():
            sorting = load_extractor(folder / "sorting.json", base_folder=folder)
        elif (folder / "sorting.pickle").exists():
            sorting = load_extractor(folder / "sorting.pickle", base_folder=folder)

    sorting_analyzer = SortingAnalyzer.create_memory(
        sorting, recording, sparsity=sparsity, return_scaled=return_scaled, rec_attributes=rec_attributes
    )

    # waveforms
    # need to concatenate all waveforms in one unique buffer
    # need to concatenate sampled_index and order it
    waveform_folder = folder / "waveforms"
    if waveform_folder.exists():

        spikes = sorting.to_spike_vector()
        random_spike_mask = np.zeros(spikes.size, dtype="bool")

        # first read all sampled_index to get the correct ordering
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            # unit_indices has dtype=[("spike_index", "int64"), ("segment_index", "int64")]
            unit_indices = np.load(waveform_folder / f"sampled_index_{unit_id}.npy")
            for segment_index in range(sorting.get_num_segments()):
                in_seg_selected = unit_indices[unit_indices["segment_index"] == segment_index]["spike_index"]
                spikes_indices = np.flatnonzero(
                    (spikes["unit_index"] == unit_index) & (spikes["segment_index"] == segment_index)
                )
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
            waveforms[:, :, : wfs.shape[2]][mask, :, :] = wfs

        ext = ComputeRandomSpikes(sorting_analyzer)
        ext.params = dict()
        ext.data = dict(random_spikes_indices=random_spikes_indices)
        sorting_analyzer.extensions["random_spikes"] = ext

        ext = ComputeWaveforms(sorting_analyzer)
        ext.params = dict(
            ms_before=params["ms_before"],
            ms_after=params["ms_after"],
            dtype=params["dtype"],
        )
        ext.data["waveforms"] = waveforms
        sorting_analyzer.extensions["waveforms"] = ext

    # templates saved dense
    # load cached templates
    templates = {}
    for mode in ("average", "std", "median", "percentile"):
        template_file = folder / f"templates_{mode}.npy"
        if template_file.is_file():
            templates[mode] = np.load(template_file)
    if len(templates) > 0:
        ext = ComputeTemplates(sorting_analyzer)
        ext.params = dict(nbefore=nbefore, nafter=nafter, operators=list(templates.keys()))
        for mode, arr in templates.items():
            ext.data[mode] = arr
        sorting_analyzer.extensions["templates"] = ext

    for old_name, new_name in old_extension_to_new_class_map.items():
        ext_folder = folder / old_name
        if not ext_folder.is_dir():
            continue
        new_class = get_extension_class(new_name)
        ext = new_class(sorting_analyzer)
        with open(ext_folder / "params.json", "r") as f:
            params = json.load(f)
        ext.params = params
        if new_name == "spike_amplitudes":
            amplitudes = []
            for segment_index in range(sorting.get_num_segments()):
                amplitudes.append(np.load(ext_folder / f"amplitude_segment_{segment_index}.npy"))
            amplitudes = np.concatenate(amplitudes)
            ext.data["amplitudes"] = amplitudes
        elif new_name == "spike_locations":
            ext.data["spike_locations"] = np.load(ext_folder / "spike_locations.npy")
        elif new_name == "amplitude_scalings":
            ext.data["amplitude_scalings"] = np.load(ext_folder / "amplitude_scalings.npy")
        elif new_name == "template_metrics":
            import pandas as pd

            ext.data["metrics"] = pd.read_csv(ext_folder / "metrics.csv", index_col=0)
        elif new_name == "template_similarity":
            ext.data["similarity"] = np.load(ext_folder / "similarity.npy")
        elif new_name == "unit_locations":
            ext.data["unit_locations"] = np.load(ext_folder / "unit_locations.npy")
        elif new_name == "correlograms":
            ext.data["ccgs"] = np.load(ext_folder / "ccgs.npy")
            ext.data["bins"] = np.load(ext_folder / "bins.npy")
        elif new_name == "isi_histograms":
            ext.data["isi_histograms"] = np.load(ext_folder / "isi_histograms.npy")
            ext.data["bins"] = np.load(ext_folder / "bins.npy")
        elif new_name == "noise_levels":
            ext.data["noise_levels"] = np.load(ext_folder / "noise_levels.npy")
        elif new_name == "quality_metrics":
            import pandas as pd

            ext.data["metrics"] = pd.read_csv(ext_folder / "metrics.csv", index_col=0)
        elif new_name == "principal_components":
            # the waveform folder is needed to get the sampled indices
            if waveform_folder.exists():
                # read params
                params_file = ext_folder / "params.json"
                with open(params_file, "r") as f:
                    params = json.load(f)
                n_components = params["n_components"]
                n_channels = len(rec_attributes["channel_ids"])
                num_spikes = len(random_spikes_indices)
                mode = params["mode"]
                if mode == "by_channel_local":
                    pc_all = np.zeros((num_spikes, n_components, n_channels), dtype=params["dtype"])
                elif mode == "by_channel_global":
                    pc_all = np.zeros((num_spikes, n_components, n_channels), dtype=params["dtype"])
                elif mode == "concatenated":
                    pc_all = np.zeros((num_spikes, n_components), dtype=params["dtype"])
                # then read pc per units
                some_spikes = spikes[random_spikes_indices]
                for unit_index, unit_id in enumerate(sorting.unit_ids):
                    pc_one = np.load(ext_folder / f"pca_{unit_id}.npy")
                    mask = some_spikes["unit_index"] == unit_index
                    pc_all[mask, ...] = pc_one
                ext.data["pca_projection"] = pc_all

        sorting_analyzer.extensions[new_name] = ext

    return sorting_analyzer


# this was never used, let's comment it out
# def _read_old_waveforms_extractor_zarr(folder, sorting):
#     import zarr

#     folder = Path(folder)
#     waveforms_root = zarr.open(folder, mode="r")

#     params = waveforms_root.attrs["params"]

#     rec_attributes = waveforms_root.get("recording_info").attrs["recording_attributes"]
#     # the probe is handle ouside the main json
#     if "probegroup" in waveforms_root.get("recording_info").attrs:
#         probegroup_dict = waveforms_root.get("recording_info").attrs["probegroup"]
#         rec_attributes["probegroup"] = probeinterface.ProbeGroup.from_dict(probegroup_dict)
#     else:
#         rec_attributes["probegroup"] = None

#     # recording
#     recording = None
#     try:
#         recording_dict = waveforms_root.attrs["recording"]
#         recording = load_extractor(recording_dict, base_folder=folder)
#     except:
#         pass

#     # sorting
#     if sorting is None:
#         assert "sorting" in waveforms_root.attrs, "Could not load sorting object"
#         sorting_dict = waveforms_root.attrs["sorting"]
#         sorting = load_extractor(sorting_dict, base_folder=folder)

#     if "sparsity" in waveforms_root.attrs:
#         sparsity_dict = waveforms_root.attrs["sparsity"]
#         sparsity = ChannelSparsity.from_dict(sparsity_dict)
#     else:
#         sparsity = None

#     sorting_analyzer = SortingAnalyzer.create_memory(sorting, recording, sparsity, rec_attributes=rec_attributes)

#     # waveforms
#     # need to concatenate all waveforms in one unique buffer
#     # need to concatenate sampled_index and order it
#     waveform_group = waveforms_root.get("waveforms", None)
#     if waveform_group:
#         spikes = sorting.to_spike_vector()
#         random_spike_mask = np.zeros(spikes.size, dtype="bool")

#         # first read all sampled_index to get the correct ordering
#         for unit_index, unit_id in enumerate(sorting.unit_ids):
#             # unit_indices has dtype=[("spike_index", "int64"), ("segment_index", "int64")]
#             unit_indices = waveform_group[f"sampled_index_{unit_id}"][:]
#             for segment_index in range(sorting.get_num_segments()):
#                 in_seg_selected = unit_indices[unit_indices["segment_index"] == segment_index]["spike_index"]
#                 spikes_indices = np.flatnonzero(
#                     (spikes["unit_index"] == unit_index) & (spikes["segment_index"] == segment_index)
#                 )
#                 random_spike_mask[spikes_indices[in_seg_selected]] = True
#         random_spikes_indices = np.flatnonzero(random_spike_mask)

#         num_spikes = random_spikes_indices.size
#         if sparsity is None:
#             max_num_channel = len(rec_attributes["channel_ids"])
#         else:
#             max_num_channel = np.max(np.sum(sparsity.mask, axis=1))

#         nbefore = int(params["ms_before"] * sorting.sampling_frequency / 1000.0)
#         nafter = int(params["ms_after"] * sorting.sampling_frequency / 1000.0)

#         waveforms = np.zeros((num_spikes, nbefore + nafter, max_num_channel), dtype=params["dtype"])
#         # then read waveforms per units
#         some_spikes = spikes[random_spikes_indices]
#         for unit_index, unit_id in enumerate(sorting.unit_ids):
#             wfs = waveform_group[f"waveforms_{unit_id}"]
#             mask = some_spikes["unit_index"] == unit_index
#             waveforms[:, :, : wfs.shape[2]][mask, :, :] = wfs

#         ext = ComputeRandomSpikes(sorting_analyzer)
#         ext.params = dict()
#         ext.data = dict(random_spikes_indices=random_spikes_indices)

#         ext = ComputeWaveforms(sorting_analyzer)
#         ext.params = dict(
#             ms_before=params["ms_before"],
#             ms_after=params["ms_after"],
#             return_scaled=params["return_scaled"],
#             dtype=params["dtype"],
#         )
#         ext.data["waveforms"] = waveforms
#         sorting_analyzer.extensions["waveforms"] = ext

#     # templates saved dense
#     # load cached templates
#     templates = {}
#     for mode in ("average", "std", "median", "percentile"):
#         template_data = waveforms_root.get(f"templates_{mode}", None)
#         if template_data:
#             templates[mode] = template_data
#     if len(templates) > 0:
#         ext = ComputeTemplates(sorting_analyzer)
#         ext.params = dict(
#             nbefore=nbefore, nafter=nafter, return_scaled=params["return_scaled"], operators=list(templates.keys())
#         )
#         for mode, arr in templates.items():
#             ext.data[mode] = arr
#         sorting_analyzer.extensions["templates"] = ext

#     for old_name, new_name in old_extension_to_new_class_map.items():
#         ext_group = waveforms_root.get(old_name, None)
#         if ext_group is None:
#             continue
#         new_class = get_extension_class(new_name)
#         ext = new_class(sorting_analyzer)
#         params = ext_group.attrs["params"]
#         ext.params = params
#         if new_name == "spike_amplitudes":
#             amplitudes = []
#             for segment_index in range(sorting.get_num_segments()):
#                 amplitudes.append(ext_group[f"amplitude_segment_{segment_index}"])
#             amplitudes = np.concatenate(amplitudes)
#             ext.data["amplitudes"] = amplitudes
#         elif new_name == "spike_locations":
#             ext.data["spike_locations"] = ext_group["spike_locations"]
#         elif new_name == "amplitude_scalings":
#             ext.data["amplitude_scalings"] = ext_group["amplitude_scalings"]
#         elif new_name == "template_metrics":
#             import xarray

#             ext_data = xarray.open_zarr(folder, group="template_metrics/metrics").to_pandas()
#             ext_data.index.rename("", inplace=True)

#             ext.data["metrics"] = ext_data
#         elif new_name == "template_similarity":
#             ext.data["similarity"] = ext_group["similarity"]
#         elif new_name == "unit_locations":
#             ext.data["unit_locations"] = ext_group["unit_locations"]
#         elif new_name == "correlograms":
#             ext.data["ccgs"] = ext_group["ccgs"]
#             ext.data["bins"] = ext_group["bins"]
#         elif new_name == "isi_histograms":
#             ext.data["isi_histograms"] = ext_group["isi_histograms"]
#             ext.data["bins"] = ext_group["bins"]
#         elif new_name == "noise_levels":
#             ext.data["noise_levels"] = ext_group["noise_levels"]
#         elif new_name == "quality_metrics":
#             import xarray

#             ext_data = xarray.open_zarr(folder, group="quality_metrics/metrics").to_pandas()
#             ext_data.index.rename("", inplace=True)

#             ext.data["metrics"] = ext_data
#         elif new_name == "principal_components":
#             # the waveform folder is needed to get the sampled indices
#             if waveform_group is not None:
#                 # read params
#                 params = ext_group.attrs["params"]
#                 n_components = params["n_components"]
#                 n_channels = len(rec_attributes["channel_ids"])
#                 num_spikes = len(random_spikes_indices)
#                 mode = params["mode"]
#                 if mode == "by_channel_local":
#                     pc_all = np.zeros((num_spikes, n_components, n_channels), dtype=params["dtype"])
#                 elif mode == "by_channel_global":
#                     pc_all = np.zeros((num_spikes, n_components, n_channels), dtype=params["dtype"])
#                 elif mode == "concatenated":
#                     pc_all = np.zeros((num_spikes, n_components), dtype=params["dtype"])
#                 # then read pc per units
#                 some_spikes = spikes[random_spikes_indices]
#                 for unit_index, unit_id in enumerate(sorting.unit_ids):
#                     pc_one = ext_group[f"pca_{unit_id}"]
#                     mask = some_spikes["unit_index"] == unit_index
#                     pc_all[mask, ...] = pc_one
#                 ext.data["pca_projection"] = pc_all

#         sorting_analyzer.extensions[new_name] = ext

#     return sorting_analyzer
