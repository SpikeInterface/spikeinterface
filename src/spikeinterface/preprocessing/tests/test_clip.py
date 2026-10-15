from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import clip, blank_saturation

import numpy as np


def test_clip(create_cache_folder):
    rec = generate_recording()

    rec0 = clip(rec, a_min=-2, a_max=3.0)
    rec0.save(folder=create_cache_folder / "rec_clip0", verbose=False)

    rec1 = clip(rec, a_min=-1.5)
    rec1.save(folder=create_cache_folder / "rec_clip1", verbose=False)

    traces0 = rec0.get_traces(segment_index=0, channel_ids=["1"])
    assert traces0.shape[1] == 1

    assert np.all(-2 <= traces0[0] <= 3)

    traces1 = rec1.get_traces(segment_index=0, channel_ids=["0", "1"])
    assert traces1.shape[1] == 2

    assert np.all(-1.5 <= traces1[1])


def test_blank_saturation(create_cache_folder):
    rec = generate_recording()

    rec0 = blank_saturation(rec, abs_threshold=3.0)
    rec0.save(folder=create_cache_folder / "rec_sat0", verbose=False)

    rec1 = blank_saturation(rec, quantile_threshold=0.01, direction="both", chunk_size=10000)
    rec1.save(folder=create_cache_folder / "rec_sat1", verbose=False)

    traces0 = rec0.get_traces(segment_index=0, channel_ids=["1"])
    assert traces0.shape[1] == 1
    assert np.all(traces0 < 3.0)

    traces1 = rec1.get_traces(segment_index=0, channel_ids=["0"])
    assert traces1.shape[1] == 1
    # use a smaller value to be sure
    a_min = rec1.segments[0].a_min
    assert np.all(traces1 >= a_min)


if __name__ == "__main__":
    import tempfile

    cache_folder = tempfile.TemporaryDirectory()
    test_clip(create_cache_folder=cache_folder.name)
    test_blank_saturation(create_cache_folder=cache_folder.name)
