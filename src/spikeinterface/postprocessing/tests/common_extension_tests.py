import pytest
import numpy as np
import pandas as pd
import shutil
import platform
from pathlib import Path

# from spikeinterface import extract_waveforms, load_extractor, load_waveforms, compute_sparsity
# from spikeinterface.core.generate import generate_ground_truth_recording

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import start_sorting_result
from spikeinterface.core import estimate_sparsity


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"

def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0, 20.0], sampling_frequency=24000.0, num_channels=10, num_units=5,
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

    This automatically precompute extension dependencies with default params before running computation.

    This also test the select_units() ability.
    """
    extension_class = None
    extension_function_kwargs_list = None
    def setUp(self):
        
        recording, sorting = get_dataset()
        # sparsity is computed once for all cases to save processing
        sparsity = estimate_sparsity(recording, sorting)

        self.sorting_results = {}
        for sparse in (True, False):
            for format in ("memory", "binary_folder", "zarr"):
                sparsity_ = sparsity if sparse else None
                sorting_result = get_sorting_result(recording, sorting, format=format, sparsity=sparsity_, name=self.extension_class.extension_name)
                key = f"spare{sparse}_{format}"
                self.sorting_results[key] = sorting_result
    
    @property
    def extension_name(self):
        return self.extension_class.extension_name

    def _check_one(self, sorting_result):
        sorting_result.select_random_spikes(max_spikes_per_unit=50, seed=2205)

        for dependency_name in self.extension_class.depend_on:
            if "|" in dependency_name:
                dependency_name = dependency_name.split("|")[0]
            sorting_result.compute(dependency_name)

        
        for kwargs in self.extension_function_kwargs_list:
            sorting_result.compute(self.extension_name, **kwargs)
        ext = sorting_result.get_extension(self.extension_name)
        assert ext is not None
        assert len(ext.data) > 0
        
        some_unit_ids = sorting_result.unit_ids[::2]
        sliced = sorting_result.select_units(some_unit_ids, format="memory")
        assert np.array_equal(sliced.unit_ids, sorting_result.unit_ids[::2])
        # print(sliced)


    def test_extension(self):

        for key, sorting_result in self.sorting_results.items():
            print()
            print(key)
            self._check_one(sorting_result)



class WaveformExtensionCommonTestSuite:
    """
    This class runs common tests for extensions.
    """

    extension_class = None
    extension_data_names = []
    extension_function_kwargs_list = None

    # this flag enables us to check that all backends have the same contents
    exact_same_content = True

    def _clean_all_folders(self):
        for name in (
            "toy_rec_1seg",
            "toy_sorting_1seg",
            "toy_waveforms_1seg",
            "toy_rec_2seg",
            "toy_sorting_2seg",
            "toy_waveforms_2seg",
            "toy_sorting_2seg.zarr",
            "toy_sorting_2seg_sparse",
        ):
            if (cache_folder / name).is_dir():
                shutil.rmtree(cache_folder / name)

        for name in ("toy_waveforms_1seg", "toy_waveforms_2seg", "toy_sorting_2seg_sparse"):
            for ext in self.extension_data_names:
                folder = self.cache_folder / f"{name}_{ext}_selected"
                if folder.exists():
                    shutil.rmtree(folder)

    def setUp(self):
        self.cache_folder = cache_folder
        self._clean_all_folders()

        # 1-segment
        recording, sorting = generate_ground_truth_recording(
            durations=[10],
            sampling_frequency=30000,
            num_channels=12,
            num_units=10,
            dtype="float32",
            seed=91,
            generate_sorting_kwargs=dict(add_spikes_on_borders=True),
            noise_kwargs=dict(noise_level=10.0, strategy="tile_pregenerated"),
        )

        # add gains and offsets and save
        gain = 0.1
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(0)

        recording = recording.save(folder=cache_folder / "toy_rec_1seg")
        sorting = sorting.save(folder=cache_folder / "toy_sorting_1seg")

        we1 = extract_waveforms(
            recording,
            sorting,
            cache_folder / "toy_waveforms_1seg",
            max_spikes_per_unit=500,
            sparse=False,
            n_jobs=1,
            chunk_size=30000,
            overwrite=True,
        )
        self.we1 = we1
        self.sparsity1 = compute_sparsity(we1, method="radius", radius_um=50)

        # 2-segments
        recording, sorting = generate_ground_truth_recording(
            durations=[10, 5],
            sampling_frequency=30000,
            num_channels=12,
            num_units=10,
            dtype="float32",
            seed=91,
            generate_sorting_kwargs=dict(add_spikes_on_borders=True),
            noise_kwargs=dict(noise_level=10.0, strategy="tile_pregenerated"),
        )
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(0)
        recording = recording.save(folder=cache_folder / "toy_rec_2seg")
        sorting = sorting.save(folder=cache_folder / "toy_sorting_2seg")

        we2 = extract_waveforms(
            recording,
            sorting,
            cache_folder / "toy_waveforms_2seg",
            max_spikes_per_unit=500,
            sparse=False,
            n_jobs=1,
            chunk_size=30000,
            overwrite=True,
        )
        self.we2 = we2

        # make we read-only
        if platform.system() != "Windows":
            we_ro_folder = cache_folder / "toy_waveforms_2seg_readonly"
            if not we_ro_folder.is_dir():
                shutil.copytree(we2.folder, we_ro_folder)
            # change permissions (R+X)
            we_ro_folder.chmod(0o555)
            self.we_ro = load_waveforms(we_ro_folder)

        self.sparsity2 = compute_sparsity(we2, method="radius", radius_um=30)
        we_memory = extract_waveforms(
            recording,
            sorting,
            mode="memory",
            sparse=False,
            max_spikes_per_unit=500,
            n_jobs=1,
            chunk_size=30000,
        )
        self.we_memory2 = we_memory

        self.we_zarr2 = we_memory.save(folder=cache_folder / "toy_sorting_2seg", overwrite=True, format="zarr")

        # use best channels for PC-concatenated
        sparsity = compute_sparsity(we_memory, method="best_channels", num_channels=2)
        self.we_sparse = we_memory.save(
            folder=cache_folder / "toy_sorting_2seg_sparse", format="binary", sparsity=sparsity, overwrite=True
        )

    def tearDown(self):
        # delete object to release memmap
        del self.we1, self.we2, self.we_memory2, self.we_zarr2, self.we_sparse
        if hasattr(self, "we_ro"):
            del self.we_ro

        # allow pytest to delete RO folder
        if platform.system() != "Windows":
            we_ro_folder = cache_folder / "toy_waveforms_2seg_readonly"
            we_ro_folder.chmod(0o777)

        self._clean_all_folders()

    def _test_extension_folder(self, we, in_memory=False):
        if self.extension_function_kwargs_list is None:
            extension_function_kwargs_list = [dict()]
        else:
            extension_function_kwargs_list = self.extension_function_kwargs_list
        for ext_kwargs in extension_function_kwargs_list:
            compute_func = self.extension_class.get_extension_function()
            _ = compute_func(we, load_if_exists=False, **ext_kwargs)

            # reload as an extension from we
            assert self.extension_class.extension_name in we.get_available_extension_names()
            assert we.has_extension(self.extension_class.extension_name)
            ext = we.load_extension(self.extension_class.extension_name)
            assert isinstance(ext, self.extension_class)
            for ext_name in self.extension_data_names:
                assert ext_name in ext._extension_data

            if not in_memory:
                ext_loaded = self.extension_class.load(we.folder, we)
                for ext_name in self.extension_data_names:
                    assert ext_name in ext_loaded._extension_data

            # test select units
            # print('test select units', we.format)
            if we.format == "binary":
                new_folder = cache_folder / f"{we.folder.stem}_{self.extension_class.extension_name}_selected"
                if new_folder.is_dir():
                    shutil.rmtree(new_folder)
                we_new = we.select_units(
                    unit_ids=we.sorting.unit_ids[::2],
                    new_folder=new_folder,
                )
                # check that extension is present after select_units()
                assert self.extension_class.extension_name in we_new.get_available_extension_names()
            elif we.folder is None:
                # test select units in-memory and zarr
                we_new = we.select_units(unit_ids=we.sorting.unit_ids[::2])
                # check that extension is present after select_units()
                assert self.extension_class.extension_name in we_new.get_available_extension_names()
            if we.format == "zarr":
                # select_units() not supported for Zarr
                pass

    def test_extension(self):
        print("Test extension", self.extension_class)
        # 1 segment
        print("1 segment", self.we1)
        self._test_extension_folder(self.we1)

        # 2 segment
        print("2 segment", self.we2)
        self._test_extension_folder(self.we2)
        # memory
        print("Memory", self.we_memory2)
        self._test_extension_folder(self.we_memory2, in_memory=True)
        # zarr
        # @alessio : this need to be fixed the PCA extention do not work wih zarr
        print("Zarr", self.we_zarr2)
        self._test_extension_folder(self.we_zarr2)

        # sparse
        print("Sparse", self.we_sparse)
        self._test_extension_folder(self.we_sparse)

        if self.exact_same_content:
            # check content is the same across modes: memory/content/zarr

            for ext in self.we2.get_available_extension_names():
                print(f"Testing data for {ext}")
                ext_memory = self.we_memory2.load_extension(ext)
                ext_folder = self.we2.load_extension(ext)
                ext_zarr = self.we_zarr2.load_extension(ext)

                for ext_data_name, ext_data_mem in ext_memory._extension_data.items():
                    ext_data_folder = ext_folder._extension_data[ext_data_name]
                    ext_data_zarr = ext_zarr._extension_data[ext_data_name]
                    if isinstance(ext_data_mem, np.ndarray):
                        np.testing.assert_array_equal(ext_data_mem, ext_data_folder)
                        np.testing.assert_array_equal(ext_data_mem, ext_data_zarr)
                    elif isinstance(ext_data_mem, pd.DataFrame):
                        assert ext_data_mem.equals(ext_data_folder)
                        assert ext_data_mem.equals(ext_data_zarr)
                    else:
                        print(f"{ext_data_name} of type {type(ext_data_mem)} not tested.")

        # read-only - Extension is memory only
        if platform.system() != "Windows":
            _ = self.extension_class.get_extension_function()(self.we_ro, load_if_exists=False)
            assert self.extension_class.extension_name in self.we_ro.get_available_extension_names()
            ext_ro = self.we_ro.load_extension(self.extension_class.extension_name)
            assert ext_ro.format == "memory"
            assert ext_ro.extension_folder is None
