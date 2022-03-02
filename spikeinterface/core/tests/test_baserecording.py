"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_raises

from probeinterface import Probe

from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording, load_extractor
from spikeinterface.core.base import BaseExtractor

if getattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BaseRecording():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'

    file_paths = [cache_folder / f'test_base_recording_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        a = np.memmap(file_paths[i], dtype=dtype,
                      mode='w+', shape=(num_samples, num_chan))
        a[:] = np.random.randn(*a.shape).astype(dtype)
    rec = BinaryRecordingExtractor(
        file_paths, sampling_frequency, num_chan, dtype)

    assert rec.get_num_segments() == 2
    assert rec.get_num_channels() == 3

    assert np.all(rec.ids_to_indices([0, 1, 2]) == [0, 1, 2])
    assert np.all(rec.ids_to_indices(
        [0, 1, 2], prefer_slice=True) == slice(0, 3, None))

    # annotations / properties
    rec.annotate(yep='yop')
    assert rec.get_annotation('yep') == 'yop'

    rec.set_channel_groups([0, 0, 1])

    rec.set_property('quality', [1., 3.3, np.nan])
    values = rec.get_property('quality')
    assert np.all(values[:2] == [1., 3.3, ])

    # missing property
    rec.set_property('string_property', ["ciao", "bello"], ids=[0, 1])
    values = rec.get_property('string_property')
    assert values[2] == ""

    # setting an different type raises an error
    assert_raises(Exception, rec.set_property, key='string_property_nan', values=["ciao", "bello"], ids=[0, 1],
                  missing_value=np.nan)

    # int properties without missing values raise an error
    assert_raises(Exception, rec.set_property,
                  key='int_property', values=[5, 6], ids=[1, 2])

    rec.set_property('int_property', [5, 6], ids=[1, 2], missing_value=200)
    values = rec.get_property('int_property')
    assert values.dtype.kind == "i"

    times0 = rec.get_times(segment_index=0)

    # dump/load dict
    d = rec.to_dict()
    rec2 = BaseExtractor.from_dict(d)
    rec3 = load_extractor(d)

    # dump/load json
    rec.dump_to_json(cache_folder / 'test_BaseRecording.json')
    rec2 = BaseExtractor.load(cache_folder / 'test_BaseRecording.json')
    rec3 = load_extractor(cache_folder / 'test_BaseRecording.json')

    # dump/load pickle
    rec.dump_to_pickle(cache_folder / 'test_BaseRecording.pkl')
    rec2 = BaseExtractor.load(cache_folder / 'test_BaseRecording.pkl')
    rec3 = load_extractor(cache_folder / 'test_BaseRecording.pkl')

    # dump/load dict - relative
    d = rec.to_dict(relative_to=cache_folder)
    rec2 = BaseExtractor.from_dict(d, base_folder=cache_folder)
    rec3 = load_extractor(d, base_folder=cache_folder)

    # dump/load json
    rec.dump_to_json(cache_folder / 'test_BaseRecording_rel.json', relative_to=cache_folder)
    rec2 = BaseExtractor.load(cache_folder / 'test_BaseRecording_rel.json', base_folder=cache_folder)
    rec3 = load_extractor(
        cache_folder / 'test_BaseRecording_rel.json', base_folder=cache_folder)

    # cache to binary
    folder = cache_folder / 'simple_recording'
    rec.save(format='binary', folder=folder)
    rec2 = BaseExtractor.load_from_folder(folder)
    assert 'quality' in rec2.get_property_keys()
    values = rec2.get_property('quality')
    assert values[0] == 1.
    assert values[1] == 3.3
    assert np.isnan(values[2])

    groups = rec2.get_channel_groups()
    assert np.array_equal(groups, [0, 0, 1])

    # but also possible
    rec3 = BaseExtractor.load(cache_folder / 'simple_recording')

    # cache to memory
    rec4 = rec3.save(format='memory')

    traces4 = rec4.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces4, traces)

    # cache joblib several jobs
    folder = cache_folder / 'simple_recording2'
    rec2 = rec.save(folder=folder, chunk_size=10, n_jobs=4)
    traces2 = rec2.get_traces(segment_index=0)

    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0., 0.], [0., 15.], [0, 30.]]
    probe.set_contacts(positions=positions, shapes='circle',
                       shape_params={'radius': 5})
    probe.set_device_channel_indices([2, -1, 0])
    probe.create_auto_shape()

    rec_p = rec.set_probe(probe, group_mode='by_shank')
    rec_p = rec.set_probe(probe, group_mode='by_probe')
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.], [0., 0.]])

    probe2 = rec_p.get_probe()
    positions3 = probe2.contact_positions
    assert np.array_equal(positions2, positions3)

    assert np.array_equal(probe2.device_channel_indices, [0, 1])

    # test save with probe
    folder = cache_folder / 'simple_recording3'
    rec2 = rec_p.save(folder=folder, chunk_size=10, n_jobs=2)
    rec2 = load_extractor(folder)
    probe2 = rec2.get_probe()
    assert np.array_equal(probe2.contact_positions, [[0, 30.], [0., 0.]])
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.], [0., 0.]])
    traces2 = rec2.get_traces(segment_index=0)
    assert np.array_equal(traces2, rec_p.get_traces(segment_index=0))

    # from probeinterface.plotting import plot_probe_group, plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe)
    # plot_probe(probe2)
    # plt.show()

    # test return_scale
    sampling_frequency = 30000
    traces = np.zeros((1000, 5), dtype='int16')
    rec_int16 = NumpyRecording([traces], sampling_frequency)
    assert rec_int16.get_dtype() == 'int16'

    traces_int16 = rec_int16.get_traces()
    assert traces_int16.dtype == 'int16'
    # return_scaled raise error when no gain_to_uV/offset_to_uV properties
    with pytest.raises(ValueError):
        traces_float32 = rec_int16.get_traces(return_scaled=True)
    rec_int16.set_property('gain_to_uV', [.195] * 5)
    rec_int16.set_property('offset_to_uV', [0.] * 5)
    traces_float32 = rec_int16.get_traces(return_scaled=True)
    assert traces_float32.dtype == 'float32'

    # test with t_start
    rec = BinaryRecordingExtractor(
        file_paths, sampling_frequency, num_chan, dtype, t_starts=np.arange(num_seg)*10.)
    times1 = rec.get_times(1)
    folder = cache_folder / 'recording_with_t_start'
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))

    # test with time_vector
    rec = BinaryRecordingExtractor(
        file_paths, sampling_frequency, num_chan, dtype)
    rec.set_times(np.arange(num_samples) /
                  sampling_frequency + 30., segment_index=0)
    rec.set_times(np.arange(num_samples) /
                  sampling_frequency + 40., segment_index=1)
    times1 = rec.get_times(1)
    folder = cache_folder / 'recording_with_times'
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))
    rec3 = load_extractor(folder)
    assert np.allclose(times1, rec3.get_times(1))


if __name__ == '__main__':
    test_BaseRecording()
