import pytest
from pathlib import Path
import shutil
import numpy as np
import platform
import zarr


from spikeinterface.core import generate_recording, generate_sorting, NumpySorting, ChannelSparsity
from spikeinterface import WaveformExtractor, BaseRecording, extract_waveforms


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_WaveformExtractor():
    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations,
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = cache_folder / "wf_rec1"
    recording = recording.save(folder=folder_rec)
    num_units = 15
    sorting = generate_sorting(num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
    recording = recording.save()
    sorting = sorting.save()


    mask = np.zeros((num_units, num_channels), dtype=bool)
    mask[:, 0] = True
    sparsity_ext = ChannelSparsity(mask, sorting.unit_ids, recording.channel_ids)

    for mode in ["folder", "memory"]:
        for sparsity in [None, sparsity_ext]:

            folder = cache_folder / 'test_waveform_extractor'
            if folder.is_dir():
                shutil.rmtree(folder)

            print(mode, sparsity)

            if mode == "memory":
                wf_folder = None
            else:
                wf_folder = folder

            we = WaveformExtractor.create(recording, sorting, wf_folder, mode=mode,
                                          sparsity=sparsity)

            we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)

            we.run_extract_waveforms(n_jobs=1, chunk_size=30000)
            we.run_extract_waveforms(n_jobs=4, chunk_size=30000, progress_bar=True)


            wfs = we.get_waveforms(0)
            assert wfs.shape[0] <= 500
            if sparsity is not None:
                assert wfs.shape[1:] == (210, 1)
            else:
                assert wfs.shape[1:] == (210, num_channels)


            wfs, sampled_index = we.get_waveforms(0, with_index=True)

            if mode == "folder":
                # load back
                we = WaveformExtractor.load(folder)
            
            if sparsity is not None:
                assert we.is_sparse()

            wfs = we.get_waveforms(0)
            if mode == "folder":
                assert isinstance(wfs, np.memmap)
            wfs_array = we.get_waveforms(0, lazy=False)
            assert isinstance(wfs_array, np.ndarray)

            
            template = we.get_template(0)
            if sparsity is None:
                assert template.shape == (210, 2)
            else:
                assert template.shape == (210, 1)
            templates = we.get_all_templates()
            assert templates.shape == (num_units, 210, num_channels)

            if sparsity is not None:
                assert np.all(templates[:, :, 1] == 0)

            template_std = we.get_template(0, mode='std')
            if sparsity is None:
                assert template_std.shape == (210, num_channels)
            else:
                assert template_std.shape == (210, 1)
            template_std = we.get_all_templates(mode='std')
            assert template_std.shape == (num_units, 210, num_channels)

            if sparsity is not None:
                assert np.all(template_std[:, :, 1] == 0)

            template_segment = we.get_template_segment(unit_id=0, segment_index=0)
            if sparsity is None:
                assert template_segment.shape == (210, num_channels)
            else:
                assert template_segment.shape == (210, 1)

            # test filter units
            keep_units = sorting.get_unit_ids()[::2]
            if (cache_folder / "we_filt").is_dir():
                shutil.rmtree(cache_folder / "we_filt")
            wf_filt = we.select_units(keep_units, cache_folder / "we_filt")
            for unit in wf_filt.sorting.get_unit_ids():
                assert unit in keep_units
            filtered_templates = wf_filt.get_all_templates()
            assert filtered_templates.shape == (len(keep_units), 210, num_channels)
            if sparsity is not None:
                wf_filt.is_sparse()

            # test save
            if (cache_folder / f"we_saved_{mode}").is_dir():
                shutil.rmtree(cache_folder / f"we_saved_{mode}")
            we_saved = we.save(cache_folder / f"we_saved_{mode}")
            for unit_id in we_saved.unit_ids:
                assert np.array_equal(we.get_waveforms(unit_id),
                                      we_saved.get_waveforms(unit_id))
                assert np.array_equal(we.get_sampled_indices(unit_id),
                                      we_saved.get_sampled_indices(unit_id))
                assert np.array_equal(we.get_all_templates(),
                                      we_saved.get_all_templates())
            wfs = we_saved.get_waveforms(0)
            assert isinstance(wfs, np.memmap)
            wfs_array = we_saved.get_waveforms(0, lazy=False)
            assert isinstance(wfs_array, np.ndarray)

            if (cache_folder / f"we_saved_{mode}.zarr").is_dir():
                shutil.rmtree(cache_folder / f"we_saved_{mode}.zarr")
            we_saved_zarr = we.save(cache_folder / f"we_saved_{mode}", format="zarr")
            for unit_id in we_saved_zarr.unit_ids:
                assert np.array_equal(we.get_waveforms(unit_id),
                                      we_saved_zarr.get_waveforms(unit_id))
                assert np.array_equal(we.get_sampled_indices(unit_id),
                                      we_saved_zarr.get_sampled_indices(unit_id))
                assert np.array_equal(we.get_all_templates(),
                                      we_saved_zarr.get_all_templates())
            wfs = we_saved_zarr.get_waveforms(0)
            assert isinstance(wfs, zarr.Array)
            wfs_array = we_saved_zarr.get_waveforms(0, lazy=False)
            assert isinstance(wfs_array, np.ndarray)

            # test post sparsity
            if sparsity is None:
                we.set_post_sparsity(sparsity_ext)
                template = we.get_template(0)
                assert template.shape == (210, 1)
                template_std = we.get_template(0, mode="std")
                assert template_std.shape == (210, 1)
                wfs = we.get_waveforms(0)
                assert wfs.shape[2] == 1
                all_templates = we.get_all_templates()
                assert all_templates.shape == (num_units, 210, num_channels)
                we.reset_post_sparsity()

def test_extract_waveforms():
    # 2 segments

    durations = [30, 40]
    sampling_frequency = 30000.

    recording = generate_recording(
        num_channels=2, durations=durations, sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = cache_folder / "wf_rec2"
    recording = recording.save(folder=folder_rec)
    sorting = generate_sorting(
        num_units=5, sampling_frequency=sampling_frequency, durations=durations)
    folder_sort = cache_folder / "wf_sort2"
    sorting = sorting.save(folder=folder_sort)
    # test without dump !!!!
    #  recording = recording.save()
    #  sorting = sorting.save()

    folder1 = cache_folder / 'test_extract_waveforms_1job'
    if folder1.is_dir():
        shutil.rmtree(folder1)
    we1 = extract_waveforms(recording, sorting, folder1,
                            max_spikes_per_unit=None, return_scaled=False)

    folder2 = cache_folder / 'test_extract_waveforms_2job'
    if folder2.is_dir():
        shutil.rmtree(folder2)
    we2 = extract_waveforms(recording, sorting, folder2, n_jobs=2, total_memory="10M", max_spikes_per_unit=None,
                            return_scaled=False)

    wf1 = we1.get_waveforms(0)
    wf2 = we2.get_waveforms(0)
    assert np.array_equal(wf1, wf2)

    folder3 = cache_folder / 'test_extract_waveforms_returnscaled'
    if folder3.is_dir():
        shutil.rmtree(folder3)

    # set scaling values to recording
    gain = 0.1
    recording.set_channel_gains(gain)
    recording.set_channel_offsets(0)
    we3 = extract_waveforms(recording, sorting, folder3, n_jobs=2, total_memory="10M", max_spikes_per_unit=None,
                            return_scaled=True)

    wf3 = we3.get_waveforms(0)
    assert np.array_equal((wf1).astype("float32") * gain, wf3)

    # test in memory
    we_mem = extract_waveforms(recording, sorting, folder=None, mode="memory",
                               n_jobs=2, total_memory="10M", max_spikes_per_unit=None,
                               return_scaled=True)
    wf_mem = we_mem.get_waveforms(0)
    assert np.array_equal(wf_mem, wf3)

    # Test unfiltered recording
    recording.annotate(is_filtered=False)

    folder_crash = cache_folder / "test_extract_waveforms_crash"
    with pytest.raises(Exception):
        we1 = extract_waveforms(recording, sorting, folder_crash,
                                max_spikes_per_unit=None, return_scaled=False)

    folder_unfiltered = cache_folder / "test_extract_waveforms_unfiltered"
    we1 = extract_waveforms(recording, sorting, folder_unfiltered, allow_unfiltered=True,
                            max_spikes_per_unit=None, return_scaled=False)


def test_recordingless():
    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations,
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    num_units = 15
    sorting = generate_sorting(
        num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # now save and delete saved file
    recording = recording.save(folder=cache_folder / "recording1")
    sorting = sorting.save(folder=cache_folder / "sorting1")

    # recording and sorting are not dumpable
    wf_folder = cache_folder / "wf_recordingless"

    # save with relative paths
    we = extract_waveforms(recording, sorting, wf_folder,
                           use_relative_path=True, return_scaled=False)
    we_loaded = WaveformExtractor.load(wf_folder, with_recording=False)

    assert isinstance(we.recording, BaseRecording)
    assert not we_loaded.has_recording()
    with pytest.raises(ValueError):
        # reccording cannot be accessible
        rec = we_loaded.recording
    assert we.sampling_frequency == we_loaded.sampling_frequency
    assert np.array_equal(we.recording.channel_ids, np.array(we_loaded.channel_ids))
    assert np.array_equal(we.recording.get_channel_locations(),
                          np.array(we_loaded.get_channel_locations()))
    assert we.get_num_channels() == we_loaded.get_num_channels()
    
    probe = we_loaded.get_probe()
    probegroup = we_loaded.get_probegroup()
    

    # delete original recording and rely on rec_attributes
    if platform.system() != "Windows":
        shutil.rmtree(cache_folder / "recording1")
        we_loaded = WaveformExtractor.load(wf_folder, with_recording=False)
        assert not we_loaded.has_recording()


def test_unfiltered_extraction():
    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations,
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=False)
    folder_rec = cache_folder / "wf_unfiltered"
    recording = recording.save(folder=folder_rec)
    num_units = 15
    sorting = generate_sorting(
        num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
    recording = recording.save()
    sorting = sorting.save()

    folder = cache_folder / 'test_waveform_extractor_unfiltered'
    if folder.is_dir():
        shutil.rmtree(folder)

    for mode in ["folder", "memory"]:
        if mode == "memory":
            wf_folder = None
        else:
            wf_folder = folder

        with pytest.raises(Exception):
            we = WaveformExtractor.create(recording, sorting, wf_folder, mode=mode, allow_unfiltered=False)
        if wf_folder is not None:
            shutil.rmtree(wf_folder)
        we = WaveformExtractor.create(recording, sorting, wf_folder, mode=mode, allow_unfiltered=True)

        we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)

        we.run_extract_waveforms(n_jobs=1, chunk_size=30000)
        we.run_extract_waveforms(n_jobs=4, chunk_size=30000, progress_bar=True)

        wfs = we.get_waveforms(0)
        assert wfs.shape[0] <= 500
        assert wfs.shape[1:] == (210, num_channels)

        wfs, sampled_index = we.get_waveforms(0, with_index=True)

        if mode == "folder":
            # load back
            we = WaveformExtractor.load_from_folder(folder)

        wfs = we.get_waveforms(0)

        template = we.get_template(0)
        assert template.shape == (210, 2)
        templates = we.get_all_templates()
        assert templates.shape == (num_units, 210, num_channels)

        wf_std = we.get_template(0, mode='std')
        assert wf_std.shape == (210, num_channels)
        wfs_std = we.get_all_templates(mode='std')
        assert wfs_std.shape == (num_units, 210, num_channels)

        wf_segment = we.get_template_segment(unit_id=0, segment_index=0)
        assert wf_segment.shape == (210, num_channels)
        assert wf_segment.shape == (210, num_channels)


def test_portability():
    durations = [30, 40]
    sampling_frequency = 30000.

    folder_to_move = cache_folder / "original_folder"
    if folder_to_move.is_dir():
        shutil.rmtree(folder_to_move)
    folder_to_move.mkdir()
    folder_moved = cache_folder / "moved_folder"
    if folder_moved.is_dir():
        shutil.rmtree(folder_moved)
    # folder_moved.mkdir()

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations,
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = folder_to_move / "rec"
    recording = recording.save(folder=folder_rec)
    num_units = 15
    sorting = generate_sorting(
        num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)
    folder_sort = folder_to_move / "sort"
    sorting = sorting.save(folder=folder_sort)

    wf_folder = folder_to_move / "waveform_extractor"
    if wf_folder.is_dir():
        shutil.rmtree(wf_folder)

    # save with relative paths
    we = extract_waveforms(recording, sorting, wf_folder,
                           use_relative_path=True)

    # move all to a separate folder
    shutil.copytree(folder_to_move, folder_moved)
    wf_folder_moved = folder_moved / "waveform_extractor"
    we_loaded = extract_waveforms(
        recording, sorting, wf_folder_moved, load_if_exists=True)

    assert we_loaded.recording is not None
    assert we_loaded.sorting is not None

    assert np.allclose(we.channel_ids, we_loaded.recording.channel_ids)
    assert np.allclose(we.unit_ids, we_loaded.unit_ids)

    for unit in we.unit_ids:
        wf = we.get_waveforms(unit_id=unit)
        wf_loaded = we_loaded.get_waveforms(unit_id=unit)

        assert np.allclose(wf, wf_loaded)


def test_empty_sorting():
    sf = 30000
    num_channels = 2

    recording = generate_recording(num_channels=num_channels, sampling_frequency=sf, durations=[15.32])
    sorting = NumpySorting.from_dict({}, sf)

    folder = cache_folder / "empty_sorting"
    wvf_extractor = extract_waveforms(recording, sorting, folder, allow_unfiltered=True)

    assert len(wvf_extractor.unit_ids) == 0
    assert wvf_extractor.get_all_templates().shape == (0, wvf_extractor.nsamples, num_channels)


if __name__ == '__main__':
    test_WaveformExtractor()
    # test_extract_waveforms()
    # test_sparsity()
    # test_portability()
    # test_recordingless()
