import shutil
from pathlib import Path

import pytest
import numpy as np

import probeinterface
from probeinterface import ProbeGroup, generate_linear_probe

from spikeinterface.core import ChannelSliceRecording, BinaryRecordingExtractor
from spikeinterface.core.generate import generate_recording


def test_ChannelSliceRecording(create_cache_folder):
    cache_folder = create_cache_folder

    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = "int16"

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_{i}.raw" for i in range(num_seg)]
    for i in range(num_seg):
        traces = np.memmap(file_paths[i], dtype=dtype, mode="w+", shape=(num_samples, num_chan))
        traces[:] = np.arange(3)[None, :]
    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_chan,
        dtype=dtype,
    )

    # keep original ids
    rec_sliced = ChannelSliceRecording(rec, channel_ids=[0, 2])
    assert np.all(rec_sliced.get_channel_ids() == [0, 2])
    traces = rec_sliced.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[0, 2])
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[2, 0])
    assert traces.shape[1] == 2
    assert rec_sliced.get_parent() == rec

    assert np.allclose(rec_sliced.get_times(0), rec.get_times(0))

    # with channel ids renaming
    rec_sliced2 = ChannelSliceRecording(rec, channel_ids=[0, 2], renamed_channel_ids=[3, 4])
    assert np.all(rec_sliced2.get_channel_ids() == [3, 4])
    traces = rec_sliced2.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 0)
    assert np.all(traces[:, 1] == 2)

    traces = rec_sliced2.get_traces(segment_index=1, channel_ids=[4, 3])
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 2)
    assert np.all(traces[:, 1] == 0)

    # with probe and after save()
    probe = probeinterface.generate_linear_probe(num_elec=num_chan)
    probe.set_device_channel_indices(np.arange(num_chan))
    rec.set_probe(probe)
    rec_sliced3 = ChannelSliceRecording(rec, channel_ids=[0, 2], renamed_channel_ids=[3, 4])
    probe3 = rec_sliced3.get_probe()
    locations3 = probe3.contact_positions
    folder = cache_folder / "sliced_recording"
    rec_saved = rec_sliced3.save(folder=folder, chunk_size=10, n_jobs=2)
    probe = rec_saved.get_probe()
    assert np.array_equal(locations3, rec_saved.get_channel_locations())
    traces3 = rec_saved.get_traces(segment_index=0)
    assert np.all(traces3[:, 0] == 0)
    assert np.all(traces3[:, 1] == 2)


def test_failure_with_non_unique_channel_ids():
    durations = [1.0]
    seed = 10
    rec = generate_recording(num_channels=4, durations=durations, set_probe=False, seed=seed)
    with pytest.raises(AssertionError):
        rec_sliced = ChannelSliceRecording(rec, channel_ids=["0", "1"], renamed_channel_ids=[0, 0])


def test_remove_channels():
    """
    Check that `remove_channels` returns a recording with the correct channels removed, and that
    it raises an error if non-existent channels are given.
    """
    durations = [1.0]
    seed = 1205

    # Note: generated recordings have channel ids: '0', '1', '2', '3', ...
    rec = generate_recording(num_channels=4, durations=durations, set_probe=False, seed=seed)

    rec_sliced = rec.remove_channels(remove_channel_ids=["0", "2"])
    rec_sliced_channel_ids = rec_sliced.get_channel_ids()
    assert np.all(rec_sliced_channel_ids == np.array(["1", "3"]))

    with pytest.raises(ValueError):
        rec_sliced = rec.remove_channels(remove_channel_ids=[0, "1"])


def test_select_channels_preserves_probe_metadata():
    """Regression test for #4547: select_channels must not mis-label surviving probes."""
    probe_A = generate_linear_probe(num_elec=8, ypitch=20.0)
    probe_A.annotate(name="probe_A", manufacturer="vendor_X")
    probe_A.move([0.0, 0.0])
    probe_A.set_device_channel_indices(np.arange(8))

    probe_B = generate_linear_probe(num_elec=8, ypitch=20.0)
    probe_B.annotate(name="probe_B", manufacturer="vendor_Y")
    probe_B.move([1000.0, 0.0])
    probe_B.set_device_channel_indices(np.arange(8, 16))

    probegroup = ProbeGroup()
    probegroup.add_probe(probe_A)
    probegroup.add_probe(probe_B)

    recording = generate_recording(durations=[1.0], num_channels=16, set_probe=False)
    recording.set_probegroup(probegroup)

    # Drop all of probe A, keep only probe B
    sub = recording.select_channels(recording.channel_ids[8:])

    assert sub.has_probe()
    probes = sub.get_probes()
    assert len(probes) == 1, "Only probe B should survive channel selection"
    assert probes[0].annotations.get("name") == "probe_B"
    assert probes[0].annotations.get("manufacturer") == "vendor_Y"


if __name__ == "__main__":
    test_ChannelSliceRecording()
