import pytest

import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import remove_artifacts


def test_remove_artifacts():
    # one segment only
    rec = generate_recording(durations=[10.0])
    # rec = rec.save(folder=cache_folder / "recording")
    rec.annotate(is_filtered=True)

    triggers = [15000, 30000]
    list_triggers = [triggers]

    ms = 10
    ms_frames = int(ms * rec.get_sampling_frequency() / 1000)

    traces_all_0_clean = rec.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1_clean = rec.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)

    rec_rmart = remove_artifacts(rec, triggers, ms_before=10, ms_after=10)
    traces_all_0 = rec_rmart.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_short_0 = rec_rmart.get_traces(start_frame=triggers[0] - 10, end_frame=triggers[0] + 10)
    traces_all_1 = rec_rmart.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)
    traces_short_1 = rec_rmart.get_traces(start_frame=triggers[1] - 10, end_frame=triggers[1] + 10)

    assert not np.any(traces_all_0)
    assert not np.any(traces_all_1)
    assert not np.any(traces_short_0)
    assert not np.any(traces_short_1)

    rec_rmart_mean = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="average")
    traces_all_0 = rec_rmart_mean.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)

    rec_rmart_median = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="median")
    traces_all_1 = rec_rmart_median.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)

    assert not np.allclose(traces_all_0, traces_all_0_clean)
    assert not np.allclose(traces_all_1, traces_all_1_clean)

    rec_rmart_lin = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="linear")
    traces_all_0 = rec_rmart_lin.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1 = rec_rmart_lin.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)
    assert not np.allclose(traces_all_0, traces_all_0_clean)
    assert not np.allclose(traces_all_1, traces_all_1_clean)

    rec_rmart_cub = remove_artifacts(rec, triggers, ms_before=10, ms_after=10, mode="cubic")
    traces_all_0 = rec_rmart_cub.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_all_1 = rec_rmart_cub.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)

    assert not np.allclose(traces_all_0, traces_all_0_clean)
    assert not np.allclose(traces_all_1, traces_all_1_clean)

    # test removing single samples
    rec_rmart_0 = remove_artifacts(rec, triggers, ms_before=None, ms_after=None)
    traces_first_0 = rec_rmart_0.get_traces(start_frame=triggers[0], end_frame=triggers[0] + 1).squeeze()
    traces_second_0 = rec_rmart_0.get_traces(start_frame=triggers[0], end_frame=triggers[0] + 1).squeeze()
    assert np.allclose(traces_first_0, np.zeros(rec.get_num_channels()))
    assert np.allclose(traces_second_0, np.zeros(rec.get_num_channels()))

    rec_rmart_lin = remove_artifacts(rec, triggers, ms_before=None, ms_after=None, mode="linear")
    rec_rmart_cub = remove_artifacts(rec, triggers, ms_before=None, ms_after=None, mode="cubic")

    # test removing several artifact types at once
    triggers = [15000, 20000, 25000, 30000]
    labels = ["stim_1", "stim_2", "stim_1", "stim_2"]
    list_triggers = [triggers]
    list_labels = [labels]
    rec_rmart = remove_artifacts(rec, list_triggers, ms_before=10, ms_after=10, mode="median", list_labels=list_labels)
    rec_rmart = remove_artifacts(
        rec, list_triggers, ms_before=10, ms_after=10, mode="median", list_labels=list_labels, time_jitter=0.2
    )
    rec_rmart = remove_artifacts(
        rec,
        list_triggers,
        ms_before=10,
        ms_after=10,
        mode="median",
        list_labels=list_labels,
        time_jitter=0.2,
        scale_amplitude=True,
    )


def test_remove_artifacts_sparsity():
    # non-regression test for issue #3290: "median"/"average" modes with sparsity
    # used to raise "shapes not aligned" because the template was kept full-channel
    # while the traces were sparsified before the np.dot / subtraction.
    rec = generate_recording(num_channels=4, durations=[10.0], seed=0)
    rec.annotate(is_filtered=True)

    ms = 10
    ms_frames = int(ms * rec.get_sampling_frequency() / 1000)

    # two artifact labels, each removed on a different subset of channels
    triggers = [15000, 30000, 45000, 60000]
    labels = [0, 1, 0, 1]
    list_triggers = [triggers]
    list_labels = [labels]
    sparsity = {
        0: np.array([True, False, True, False]),
        1: np.array([False, True, False, True]),
    }

    for mode in ("median", "average"):
        rec_rmart = remove_artifacts(
            rec,
            list_triggers,
            ms_before=ms,
            ms_after=ms,
            mode=mode,
            list_labels=list_labels,
            sparsity=sparsity,
        )
        for trig, label in zip(triggers, labels):
            mask = sparsity[label]
            traces_clean = rec.get_traces(start_frame=trig - ms_frames, end_frame=trig + ms_frames)
            # get_traces must not raise (the bug reported in issue #3290)
            traces = rec_rmart.get_traces(start_frame=trig - ms_frames, end_frame=trig + ms_frames)
            # channels outside the sparsity mask are left untouched
            assert np.array_equal(traces[:, ~mask], traces_clean[:, ~mask])
            # channels inside the sparsity mask have the artifact subtracted
            assert not np.allclose(traces[:, mask], traces_clean[:, mask])


if __name__ == "__main__":
    test_remove_artifacts()
    test_remove_artifacts_sparsity()
