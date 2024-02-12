from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import shutil
import platform
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import start_sorting_result
from spikeinterface.core import estimate_sparsity


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"

def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[15.0, 5.0], sampling_frequency=24000.0, num_channels=6, num_units=3,
        generate_sorting_kwargs=dict(firing_rates=3.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params_range=dict(
                alpha=(9_000.0, 12_000.0),
            )
        ),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting

def get_sorting_result(recording, sorting, format="memory", sparsity=None, name=""):
    sparse = sparsity is not None
    if format == "memory":
        folder = None
    elif format == "binary_folder":
        folder = cache_folder / f"test_{name}_sparse{sparse}_{format}"
    elif format == "zarr":
        folder = cache_folder / f"test_{name}_sparse{sparse}_{format}.zarr"
    if folder and folder.exists():
        shutil.rmtree(folder)
    
    sortres = start_sorting_result(sorting, recording, format=format, folder=folder, sparse=False, sparsity=sparsity)

    return sortres

class ResultExtensionCommonTestSuite:
    """
    Common tests with class approach to compute extension on several cases (3 format x 2 sparsity)

    This is done a a list of differents parameters (extension_function_params_list).

    This automatically precompute extension dependencies with default params before running computation.

    This also test the select_units() ability.
    """
    extension_class = None
    extension_function_params_list = None

    @classmethod
    def setUpClass(cls):
        cls.recording, cls.sorting = get_dataset()
        # sparsity is computed once for all cases to save processing time
        cls.sparsity = estimate_sparsity(cls.recording, cls.sorting)

    # def tearDown(self):
    #     for k in list(self.sorting_results.keys()):
    #         sorting_result = self.sorting_results.pop(k)
    #         if sorting_result.format != "memory":
    #             folder = sorting_result.folder
    #             del sorting_result
    #             shutil.rmtree(folder)

    @property
    def extension_name(self):
        return self.extension_class.extension_name
    
    def _prepare_sorting_result(self, format, sparse):
        # prepare a SortingResult object with depencies already computed
        sparsity_ = self.sparsity if sparse else None
        sorting_result = get_sorting_result(self.recording, self.sorting, format=format, sparsity=sparsity_, name=self.extension_class.extension_name)
        sorting_result.select_random_spikes(max_spikes_per_unit=50, seed=2205)
        for dependency_name in self.extension_class.depend_on:
            if "|" in dependency_name:
                dependency_name = dependency_name.split("|")[0]
            sorting_result.compute(dependency_name)
        return sorting_result

    def _check_one(self, sorting_result):
        if self.extension_class.need_job_kwargs:
            job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
        else:
            job_kwargs = dict()

        for params in self.extension_function_params_list:
            print('  params', params)
            ext = sorting_result.compute(self.extension_name, **params, **job_kwargs)
            assert len(ext.data) > 0
            main_data = ext.get_data()

        ext = sorting_result.get_extension(self.extension_name)
        assert ext is not None
        
        some_unit_ids = sorting_result.unit_ids[::2]
        sliced = sorting_result.select_units(some_unit_ids, format="memory")
        assert np.array_equal(sliced.unit_ids, sorting_result.unit_ids[::2])
        # print(sliced)


    def test_extension(self):
        for sparse in (True, False):
            for format in ("memory", "binary_folder", "zarr"):
                print()
                print("sparse", sparse, format)
                sorting_result = self._prepare_sorting_result(format, sparse)
                self._check_one(sorting_result)
