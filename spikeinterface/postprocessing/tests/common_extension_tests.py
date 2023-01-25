import pytest
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from spikeinterface import extract_waveforms, load_extractor, compute_sparsity
from spikeinterface.extractors import toy_example

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"

class WaveformExtensionCommonTestSuite:
    """
    This class runs common tests for extensions.
    """
    extension_class = None
    extension_data_names = []
    extension_function_kwargs_list = None

    def setUp(self):
        self.cache_folder  =cache_folder

        # 1-segment
        recording, sorting = toy_example(
            num_segments=1, num_units=10, num_channels=12)
        gain = 0.1
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(0)
        if (cache_folder / 'toy_rec_1seg').is_dir():
            recording = load_extractor(cache_folder / 'toy_rec_1seg')
        else:
            recording = recording.save(folder=cache_folder / 'toy_rec_1seg')
        if (cache_folder / 'toy_sorting_1seg').is_dir():
            sorting = load_extractor(cache_folder / 'toy_sorting_1seg')
        else:
            sorting = sorting.save(folder=cache_folder / 'toy_sorting_1seg')
        we1 = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms_1seg',
                                ms_before=3., ms_after=4., max_spikes_per_unit=500,
                                n_jobs=1, chunk_size=30000, overwrite=True)
        self.we1 = we1
        self.sparsity1 = compute_sparsity(we1, method="radius", radius_um=50)

        # 2-segments
        recording, sorting = toy_example(num_segments=2, num_units=10)
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(0)
        if (cache_folder / 'toy_rec_2seg').is_dir():
            recording = load_extractor(cache_folder / 'toy_rec_2seg')
        else:
            recording = recording.save(folder=cache_folder / 'toy_rec_2seg')
        if (cache_folder / 'toy_sorting_2seg').is_dir():
            sorting = load_extractor(cache_folder / 'toy_sorting_2seg')
        else:
            sorting = sorting.save(folder=cache_folder / 'toy_sorting_2seg')
        we2 = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms_2seg',
                                ms_before=3., ms_after=4., max_spikes_per_unit=500,
                                n_jobs=1, chunk_size=30000, overwrite=True)
        self.we2 = we2
        self.sparsity2 = compute_sparsity(we2, method="radius", radius_um=30)
        we_memory = extract_waveforms(recording, sorting, mode="memory",
                                      ms_before=3., ms_after=4., max_spikes_per_unit=500,
                                      n_jobs=1, chunk_size=30000)
        self.we_memory2 = we_memory

        self.we_zarr2 = we_memory.save(folder=cache_folder / 'toy_sorting_2seg',
                                       overwrite=True, format="zarr")
        
        # use best channels for PC-concatenated
        sparsity = compute_sparsity(we_memory, method="best_channels", num_channels=2)
        self.we_sparse = we_memory.save(folder=cache_folder / 'toy_sorting_2seg_sparse', format="binary",
                                        sparsity=sparsity, overwrite=True)

    def _test_extension_folder(self, we, in_memory=False):
        if self.extension_function_kwargs_list is None:
            extension_function_kwargs_list = [dict()]
        else:
            extension_function_kwargs_list = self.extension_function_kwargs_list

        for ext_kwargs in extension_function_kwargs_list:
            #~ print(ext_kwargs)
            _ = self.extension_class.get_extension_function()(we, load_if_exists=False, **ext_kwargs)
            
            # reload as an extension from we
            assert self.extension_class.extension_name in we.get_available_extension_names()
            assert we.is_extension(self.extension_class.extension_name)
            ext = we.load_extension(self.extension_class.extension_name)
            assert isinstance(ext, self.extension_class)
            for ext_name in self.extension_data_names:
                assert ext_name in ext._extension_data
            if not in_memory:
                ext_loaded = self.extension_class.load(we.folder)
                for ext_name in self.extension_data_names:
                    assert ext_name in ext_loaded._extension_data
            
            # test select units
            # print('test select units', we.format)
            if we.format == "binary":
                new_folder = cache_folder / f"{we.folder.stem}_{self.extension_class.extension_name}_selected"
                if new_folder.is_dir():
                    shutil.rmtree(new_folder)
                we_new = we.select_units(unit_ids=we.sorting.unit_ids[::2], 
                                         new_folder=cache_folder / \
                                             f"{we.folder.stem}_{self.extension_class.extension_name}_selected")
                 # check that extension is present after select_units()
                assert self.extension_class.extension_name in we_new.get_available_extension_names()
            elif we.folder is None:
                # test select units in-memory and zarr
                we_new = we.select_units(unit_ids=we.sorting.unit_ids[::2])
                # check that extension is present after select_units()
                assert self.extension_class.extension_name in we_new.get_available_extension_names()
            else:
                print("select_units() not supported for Zarr")

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
        # @alessio : this need to be fixed the PCA extention do not work wih zarr
        print("Zarr", self.we_zarr2)
        self._test_extension_folder(self.we_zarr2)
        
        # sparse
        print("Sparse", self.we_sparse)
        self._test_extension_folder(self.we_sparse)

        # test content of memory/content/zarr
        for ext in self.we2.get_available_extension_names():
            print(f"Testing data for {ext}")
            ext_memory = self.we2.load_extension(ext)
            ext_folder = self.we2.load_extension(ext)
            ext_zarr = self.we2.load_extension(ext)

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
                    
