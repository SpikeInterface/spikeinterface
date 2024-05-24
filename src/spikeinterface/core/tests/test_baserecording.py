"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""

import json
import pickle
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_raises

from probeinterface import Probe

from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording, load_extractor, get_default_zarr_compressor
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.testing import check_recordings_equal

from spikeinterface.core import generate_recording

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"
    cache_folder.mkdir(exist_ok=True, parents=True)


def test_BaseRecording():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = "int16"

    file_paths = [cache_folder / f"test_base_recording_{i}.raw" for i in range(num_seg)]
    for i in range(num_seg):
        a = np.memmap(file_paths[i], dtype=dtype, mode="w+", shape=(num_samples, num_chan))
        a[:] = np.random.randn(*a.shape).astype(dtype)
    rec = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_chan, dtype=dtype
    )

    assert rec.get_num_segments() == 2
    assert rec.get_num_channels() == 3

    assert np.all(rec.ids_to_indices([0, 1, 2]) == [0, 1, 2])
    assert np.all(rec.ids_to_indices([0, 1, 2], prefer_slice=True) == slice(0, 3, None))

    # annotations / properties
    rec.annotate(yep="yop")
    assert rec.get_annotation("yep") == "yop"

    rec.set_channel_groups([0, 0, 1])

    rec.set_property("quality", [1.0, 3.3, np.nan])
    values = rec.get_property("quality")
    assert np.all(
        values[:2]
        == [
            1.0,
            3.3,
        ]
    )

    # missing property
    rec.set_property("string_property", ["ciao", "bello"], ids=[0, 1])
    values = rec.get_property("string_property")
    assert values[2] == ""

    # setting an different type raises an error
    assert_raises(
        Exception,
        rec.set_property,
        key="string_property_nan",
        values=["ciao", "bello"],
        ids=[0, 1],
        missing_value=np.nan,
    )

    # int properties without missing values raise an error
    assert_raises(Exception, rec.set_property, key="int_property", values=[5, 6], ids=[1, 2])

    rec.set_property("int_property", [5, 6], ids=[1, 2], missing_value=200)
    values = rec.get_property("int_property")
    assert values.dtype.kind == "i"

    times0 = rec.get_times(segment_index=0)

    # dump/load dict
    d = rec.to_dict(include_annotations=True, include_properties=True)
    rec2 = BaseExtractor.from_dict(d)
    rec3 = load_extractor(d)
    check_recordings_equal(rec, rec2, return_scaled=False, check_annotations=True, check_properties=True)
    check_recordings_equal(rec, rec3, return_scaled=False, check_annotations=True, check_properties=True)

    # dump/load json
    rec.dump_to_json(cache_folder / "test_BaseRecording.json")
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording.json")
    rec3 = load_extractor(cache_folder / "test_BaseRecording.json")
    check_recordings_equal(rec, rec2, return_scaled=False, check_annotations=True, check_properties=False)
    check_recordings_equal(rec, rec3, return_scaled=False, check_annotations=True, check_properties=False)

    # dump/load pickle
    rec.dump_to_pickle(cache_folder / "test_BaseRecording.pkl")
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording.pkl")
    rec3 = load_extractor(cache_folder / "test_BaseRecording.pkl")
    check_recordings_equal(rec, rec2, return_scaled=False, check_annotations=True, check_properties=True)
    check_recordings_equal(rec, rec3, return_scaled=False, check_annotations=True, check_properties=True)

    # dump/load dict - relative
    d = rec.to_dict(relative_to=cache_folder, recursive=True)
    rec2 = BaseExtractor.from_dict(d, base_folder=cache_folder)
    rec3 = load_extractor(d, base_folder=cache_folder)

    # dump/load json - relative to
    rec.dump_to_json(cache_folder / "test_BaseRecording_rel.json", relative_to=cache_folder)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel.json", base_folder=cache_folder)
    rec3 = load_extractor(cache_folder / "test_BaseRecording_rel.json", base_folder=cache_folder)

    # dump/load relative=True
    rec.dump_to_json(cache_folder / "test_BaseRecording_rel_true.json", relative_to=True)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel_true.json", base_folder=True)
    rec3 = load_extractor(cache_folder / "test_BaseRecording_rel_true.json", base_folder=True)
    check_recordings_equal(rec, rec2, return_scaled=False, check_annotations=True)
    check_recordings_equal(rec, rec3, return_scaled=False, check_annotations=True)
    with open(cache_folder / "test_BaseRecording_rel_true.json") as json_file:
        data = json.load(json_file)
        assert (
            "/" not in data["kwargs"]["file_paths"][0]
        )  # Relative to parent folder, so there shouldn't be any '/' in the path.

    # dump/load pkl - relative to
    rec.dump_to_pickle(cache_folder / "test_BaseRecording_rel.pkl", relative_to=cache_folder)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel.pkl", base_folder=cache_folder)
    rec3 = load_extractor(cache_folder / "test_BaseRecording_rel.pkl", base_folder=cache_folder)

    # dump/load relative=True
    rec.dump_to_pickle(cache_folder / "test_BaseRecording_rel_true.pkl", relative_to=True)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel_true.pkl", base_folder=True)
    rec3 = load_extractor(cache_folder / "test_BaseRecording_rel_true.pkl", base_folder=True)
    check_recordings_equal(rec, rec2, return_scaled=False, check_annotations=True)
    check_recordings_equal(rec, rec3, return_scaled=False, check_annotations=True)
    with open(cache_folder / "test_BaseRecording_rel_true.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)
        assert (
            "/" not in data["kwargs"]["file_paths"][0]
        )  # Relative to parent folder, so there shouldn't be any '/' in the path.

    # cache to binary
    folder = cache_folder / "simple_recording"
    rec.save(format="binary", folder=folder)
    rec2 = BaseExtractor.load_from_folder(folder)
    assert "quality" in rec2.get_property_keys()
    values = rec2.get_property("quality")
    assert values[0] == 1.0
    assert values[1] == 3.3
    assert np.isnan(values[2])

    groups = rec2.get_channel_groups()
    assert np.array_equal(groups, [0, 0, 1])

    # but also possible
    rec3 = BaseExtractor.load(cache_folder / "simple_recording")

    # cache to memory
    rec4 = rec3.save(format="memory", shared=False)
    traces4 = rec4.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces4, traces)

    # cache to sharedmemory
    rec5 = rec3.save(format="memory", shared=True)
    traces5 = rec5.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces5, traces)

    # cache joblib several jobs
    folder = cache_folder / "simple_recording2"
    rec2 = rec.save(format="binary", folder=folder, chunk_size=10, n_jobs=4)
    traces2 = rec2.get_traces(segment_index=0)

    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0.0, 0.0], [0.0, 15.0], [0, 30.0]]
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices([2, -1, 0])
    probe.create_auto_shape()

    rec_p = rec.set_probe(probe, group_mode="by_shank")
    rec_p = rec.set_probe(probe, group_mode="by_probe")
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])

    probe2 = rec_p.get_probe()
    positions3 = probe2.contact_positions
    assert np.array_equal(positions2, positions3)

    assert np.array_equal(probe2.device_channel_indices, [0, 1])

    # test save with probe
    folder = cache_folder / "simple_recording3"
    rec2 = rec_p.save(folder=folder, chunk_size=10, n_jobs=2)
    rec2 = load_extractor(folder)
    probe2 = rec2.get_probe()
    assert np.array_equal(probe2.contact_positions, [[0, 30.0], [0.0, 0.0]])
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])
    traces2 = rec2.get_traces(segment_index=0)
    assert np.array_equal(traces2, rec_p.get_traces(segment_index=0))

    # from probeinterface.plotting import plot_probe_group, plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe)
    # plot_probe(probe2)
    # plt.show()

    # set unconnected probe
    probe = Probe(ndim=2)
    positions = [[0.0, 0.0], [0.0, 15.0], [0, 30.0]]
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices([-1, -1, -1])
    probe.create_auto_shape()

    rec_empty_probe = rec.set_probe(probe, group_mode="by_shank")
    assert rec_empty_probe.channel_ids.size == 0

    # test return_scale
    sampling_frequency = 30000
    traces = np.zeros((1000, 5), dtype="int16")
    rec_int16 = NumpyRecording([traces], sampling_frequency)
    assert rec_int16.get_dtype() == "int16"

    traces = np.zeros((1000, 5), dtype="uint16")
    rec_uint16 = NumpyRecording([traces], sampling_frequency)
    assert rec_uint16.get_dtype() == "uint16"

    traces_int16 = rec_int16.get_traces()
    assert traces_int16.dtype == "int16"
    # return_scaled raise error when no gain_to_uV/offset_to_uV properties
    with pytest.raises(ValueError):
        traces_float32 = rec_int16.get_traces(return_scaled=True)
    rec_int16.set_property("gain_to_uV", [0.195] * 5)
    rec_int16.set_property("offset_to_uV", [0.0] * 5)
    traces_float32 = rec_int16.get_traces(return_scaled=True)
    assert traces_float32.dtype == "float32"

    # test cast unsigned
    tr_u = rec_uint16.get_traces(cast_unsigned=False)
    assert tr_u.dtype.kind == "u"
    tr_i = rec_uint16.get_traces(cast_unsigned=True)
    assert tr_i.dtype.kind == "i"
    folder = cache_folder / "recording_unsigned"
    rec_u = rec_uint16.save(folder=folder)
    rec_u.get_dtype() == "uint16"
    folder = cache_folder / "recording_signed"
    rec_i = rec_uint16.save(folder=folder, dtype="int16")
    rec_i.get_dtype() == "int16"
    assert np.allclose(
        rec_u.get_traces(cast_unsigned=False).astype("float") - (2**15), rec_i.get_traces().astype("float")
    )
    assert np.allclose(rec_u.get_traces(cast_unsigned=True), rec_i.get_traces().astype("float"))

    # test cast with dtype
    rec_float32 = rec_int16.astype("float32")
    assert rec_float32.get_dtype() == "float32"
    assert np.dtype(rec_float32.get_traces().dtype) == np.float32

    # test with t_start
    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_chan,
        dtype=dtype,
        t_starts=np.arange(num_seg) * 10.0,
    )
    times1 = rec.get_times(1)
    folder = cache_folder / "recording_with_t_start"
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))

    # test with time_vector
    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_chan,
        dtype=dtype,
    )
    rec.set_times(np.arange(num_samples) / sampling_frequency + 30.0, segment_index=0)
    rec.set_times(np.arange(num_samples) / sampling_frequency + 40.0, segment_index=1)
    times1 = rec.get_times(1)
    folder = cache_folder / "recording_with_times"
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))
    rec3 = load_extractor(folder)
    assert np.allclose(times1, rec3.get_times(1))

    # test 3d probe
    rec_3d = generate_recording(ndim=3, num_channels=30)
    locations_3d = rec_3d.get_property("location")

    locations_xy = rec_3d.get_channel_locations(axes="xy")
    assert np.allclose(locations_xy, locations_3d[:, [0, 1]])

    locations_xz = rec_3d.get_channel_locations(axes="xz")
    assert np.allclose(locations_xz, locations_3d[:, [0, 2]])

    locations_zy = rec_3d.get_channel_locations(axes="zy")
    assert np.allclose(locations_zy, locations_3d[:, [2, 1]])

    locations_xzy = rec_3d.get_channel_locations(axes="xzy")
    assert np.allclose(locations_xzy, locations_3d[:, [0, 2, 1]])

    rec_2d = rec_3d.planarize(axes="zy")
    assert np.allclose(rec_2d.get_channel_locations(), locations_3d[:, [2, 1]])

    # test save to zarr
    compressor = get_default_zarr_compressor()
    rec_zarr = rec2.save(format="zarr", folder=cache_folder / "recording", compressor=compressor)
    rec_zarr_loaded = load_extractor(cache_folder / "recording.zarr")
    # annotations is False because Zarr adds compression ratios
    check_recordings_equal(rec2, rec_zarr, return_scaled=False, check_annotations=False, check_properties=True)
    check_recordings_equal(
        rec_zarr, rec_zarr_loaded, return_scaled=False, check_annotations=False, check_properties=True
    )
    for annotation_name in rec2.get_annotation_keys():
        assert rec2.get_annotation(annotation_name) == rec_zarr.get_annotation(annotation_name)
        assert rec2.get_annotation(annotation_name) == rec_zarr_loaded.get_annotation(annotation_name)

    rec_zarr2 = rec2.save(
        format="zarr", folder=cache_folder / "recording_channel_chunk", compressor=compressor, channel_chunk_size=2
    )
    rec_zarr2_loaded = load_extractor(cache_folder / "recording_channel_chunk.zarr")

    # annotations is False because Zarr adds compression ratios
    check_recordings_equal(rec2, rec_zarr2, return_scaled=False, check_annotations=False, check_properties=True)
    check_recordings_equal(
        rec_zarr2, rec_zarr2_loaded, return_scaled=False, check_annotations=False, check_properties=True
    )
    for annotation_name in rec2.get_annotation_keys():
        assert rec2.get_annotation(annotation_name) == rec_zarr2.get_annotation(annotation_name)
        assert rec2.get_annotation(annotation_name) == rec_zarr2_loaded.get_annotation(annotation_name)

    # test cast unsigned
    rec_u = rec_uint16.save(format="zarr", folder=cache_folder / "rec_u")
    rec_u.get_dtype() == "uint16"
    rec_i = rec_uint16.save(format="zarr", folder=cache_folder / "rec_i", dtype="int16")
    rec_i.get_dtype() == "int16"
    assert np.allclose(
        rec_u.get_traces(cast_unsigned=False).astype("float") - (2**15), rec_i.get_traces().astype("float")
    )
    assert np.allclose(rec_u.get_traces(cast_unsigned=True), rec_i.get_traces().astype("float"))


def test_rename_channels():
    recording = generate_recording(durations=[1.0], num_channels=3)
    renamed_recording = recording.rename_channels(new_channel_ids=["a", "b", "c"])
    renamed_channel_ids = renamed_recording.get_channel_ids()
    assert np.array_equal(renamed_channel_ids, ["a", "b", "c"])


def test_select_channels():
    recording = generate_recording(durations=[1.0], num_channels=3)
    renamed_recording = recording.rename_channels(new_channel_ids=["a", "b", "c"])
    selected_recording = renamed_recording.select_channels(channel_ids=["a", "c"])
    selected_channel_ids = selected_recording.get_channel_ids()
    assert np.array_equal(selected_channel_ids, ["a", "c"])


if __name__ == "__main__":
    test_BaseRecording()
