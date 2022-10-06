import pytest
from spikeinterface import extract_waveforms, load_extractor
from spikeinterface.extractors import toy_example
from spikeinterface.postprocessing import get_template_channel_sparsity
from pathlib import Path

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
        self.sparsity1 = get_template_channel_sparsity(we1, method="radius",
                                                       radius_um=30)

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
        self.sparsity2 = get_template_channel_sparsity(we1, method="radius",
                                                       radius_um=30)
        we_memory = extract_waveforms(recording, sorting, mode="memory",
                                      ms_before=3., ms_after=4., max_spikes_per_unit=500,
                                      n_jobs=1, chunk_size=30000)
        self.we_memory = we_memory
        

    def _test_extension_folder(self, we, in_memory=False):
        if self.extension_function_kwargs_list is None:
            extension_function_kwargs_list = [dict()]
        else:
            extension_function_kwargs_list = self.extension_function_kwargs_list

        for ext_kwargs in extension_function_kwargs_list:
            _ = self.extension_class.get_extension_function()(we, **ext_kwargs)
            # reload as an extension from we
            assert self.extension_class in we.get_available_extensions()
            assert self.we1.is_extension(self.extension_class.extension_name)
            ext = self.we1.load_extension(self.extension_class.extension_name)
            assert isinstance(ext, self.extension_class)
            for ext_name in self.extension_data_names:
                assert ext_name in ext._extension_data
            if not in_memory:
                ext_loaded = self.extension_class.load_from_folder(we.folder)
                for ext_name in self.extension_data_names:
                    assert ext_name in ext_loaded._extension_data
    
    def test_extension(self):
        print("Test extension", self.extension_class)
        # 1 segment
        print("1 segment", self.we1)
        self._test_extension_folder(self.we1)
        # 2 segment
        print("2 segment", self.we2)
        self._test_extension_folder(self.we2)
        # memory
        print("Memory", self.we_memory)
        self._test_extension_folder(self.we_memory, in_memory=True)
