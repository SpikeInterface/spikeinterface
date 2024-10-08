"""
test for BaseSnippets are done with NumpySnippets.
but check only for BaseRecording general methods.
"""

from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_raises

from probeinterface import Probe
from spikeinterface.core import generate_snippets
from spikeinterface.core import NumpySnippets, load_extractor
from spikeinterface.core.npysnippetsextractor import NpySnippetsExtractor
from spikeinterface.core.base import BaseExtractor


def test_BaseSnippets(create_cache_folder):
    cache_folder = create_cache_folder
    duration = [4, 3]
    num_channels = 3
    nbefore = 20
    nafter = 44

    nse, sorting = generate_snippets(
        durations=duration, num_channels=num_channels, nbefore=nbefore, nafter=nafter, wf_folder=None
    )
    num_seg = len(duration)
    file_paths = [cache_folder / f"test_base_snippets_{i}.npy" for i in range(num_seg)]

    NpySnippetsExtractor.write_snippets(nse, file_paths)
    snippets = NpySnippetsExtractor(
        file_paths,
        sampling_frequency=nse.get_sampling_frequency(),
        channel_ids=nse.channel_ids,
        nbefore=nse.nbefore,
        gain_to_uV=nse.get_property("gain_to_uV"),
        offset_to_uV=nse.get_property("offset_to_uV"),
    )

    assert snippets.get_num_segments() == len(duration)
    assert snippets.get_num_channels() == num_channels

    assert np.all(snippets.ids_to_indices([0, 1, 2]) == [0, 1, 2])
    assert np.all(snippets.ids_to_indices([0, 1, 2], prefer_slice=True) == slice(0, 3, None))

    # annotations / properties
    snippets.annotate(gre="ta")
    assert snippets.get_annotation("gre") == "ta"

    snippets.set_channel_groups([0, 0, 1])
    snippets.set_property("quality", [1.0, 3.3, np.nan])
    values = snippets.get_property("quality")
    assert np.all(
        values[:2]
        == [
            1.0,
            3.3,
        ]
    )

    # missing property
    snippets.set_property("string_property", ["ciao", "bello"], ids=[0, 1])
    values = snippets.get_property("string_property")
    assert values[2] == ""

    # setting an different type raises an error
    assert_raises(
        Exception,
        snippets.set_property,
        key="string_property_nan",
        values=["hola", "chabon"],
        ids=[0, 1],
        missing_value=np.nan,
    )

    # int properties without missing values raise an error
    assert_raises(Exception, snippets.set_property, key="int_property", values=[5, 6], ids=[1, 2])

    snippets.set_property("int_property", [5, 6], ids=[1, 2], missing_value=200)
    values = snippets.get_property("int_property")
    assert values.dtype.kind == "i"

    times0 = snippets.get_frames(segment_index=0)

    seg0_times = sorting.to_spike_vector(concatenated=False)[0]["sample_index"]

    assert np.array_equal(seg0_times, times0)

    # dump/load dict
    d = snippets.to_dict()
    snippets2 = BaseExtractor.from_dict(d)
    snippets3 = load_extractor(d)

    # dump/load json
    snippets.dump_to_json(cache_folder / "test_BaseSnippets.json")
    snippets2 = BaseExtractor.load(cache_folder / "test_BaseSnippets.json")
    snippets3 = load_extractor(cache_folder / "test_BaseSnippets.json")

    # dump/load pickle
    snippets.dump_to_pickle(cache_folder / "test_BaseSnippets.pkl")
    snippets2 = BaseExtractor.load(cache_folder / "test_BaseSnippets.pkl")
    snippets3 = load_extractor(cache_folder / "test_BaseSnippets.pkl")

    # dump/load dict - relative
    d = snippets.to_dict(relative_to=cache_folder, recursive=True)
    snippets2 = BaseExtractor.from_dict(d, base_folder=cache_folder)
    snippets3 = load_extractor(d, base_folder=cache_folder)

    # dump/load json
    snippets.dump_to_json(cache_folder / "test_BaseSnippets_rel.json", relative_to=cache_folder)
    snippets2 = BaseExtractor.load(cache_folder / "test_BaseSnippets_rel.json", base_folder=cache_folder)
    snippets3 = load_extractor(cache_folder / "test_BaseSnippets_rel.json", base_folder=cache_folder)

    # cache to npy
    folder = cache_folder / "simple_snippets"
    snippets.save(format="npy", folder=folder)
    snippets2 = BaseExtractor.load_from_folder(folder)
    assert "quality" in snippets2.get_property_keys()
    values = snippets2.get_property("quality")
    assert values[0] == 1.0
    assert values[1] == 3.3
    assert np.isnan(values[2])

    groups = snippets2.get_channel_groups()
    assert np.array_equal(groups, [0, 0, 1])

    # but also possible
    snippets3 = BaseExtractor.load(cache_folder / "simple_snippets")

    # cache to memory
    snippets4 = snippets3.save(format="memory")

    waveform4 = snippets4.get_snippets(segment_index=0)
    waveform = snippets.get_snippets(segment_index=0)
    assert np.array_equal(waveform4, waveform)

    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0.0, 0.0], [0.0, 15.0], [0, 30.0]]
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices([2, -1, 0])
    probe.create_auto_shape()

    snippets_p = snippets.set_probe(probe, group_mode="by_shank")
    snippets_p = snippets.set_probe(probe, group_mode="by_probe")
    positions2 = snippets_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])

    probe2 = snippets_p.get_probe()
    positions3 = probe2.contact_positions
    assert np.array_equal(positions2, positions3)

    assert np.array_equal(probe2.device_channel_indices, [0, 1])

    # test save with probe
    folder = cache_folder / "simple_snippets3"
    snippets2 = snippets_p.save(folder=folder)
    snippets2 = load_extractor(folder)
    probe2 = snippets2.get_probe()
    assert np.array_equal(probe2.contact_positions, [[0, 30.0], [0.0, 0.0]])
    positions2 = snippets_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])
    wavefroms2 = snippets2.get_snippets(segment_index=0)
    assert np.array_equal(wavefroms2, snippets_p.get_snippets(segment_index=0))

    # test return_scale
    sampling_frequency = 30000

    spikesframes_list = np.arange(100)
    snippets_list = np.empty((100, 64, 5), dtype="int16")

    nse_int16 = NumpySnippets(
        snippets_list=snippets_list, spikesframes_list=spikesframes_list, sampling_frequency=30000
    )

    assert nse_int16.get_dtype() == "int16"

    waveforms_int16 = nse_int16.get_snippets()
    assert waveforms_int16.dtype == "int16"

    # return_scaled raise error when no gain_to_uV/offset_to_uV properties
    with pytest.raises(ValueError):
        waveforms_float32 = nse_int16.get_snippets(return_scaled=True)
    nse_int16.set_property("gain_to_uV", [0.195] * 5)
    nse_int16.set_property("offset_to_uV", [0.0] * 5)
    waveforms_float32 = nse_int16.get_snippets(return_scaled=True)
    assert waveforms_float32.dtype == "float32"


if __name__ == "__main__":
    test_BaseSnippets()
