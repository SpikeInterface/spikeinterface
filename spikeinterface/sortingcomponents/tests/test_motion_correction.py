import pytest
from pathlib import Path
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.motion_correction import correct_motion_on_peaks, correct_motion_on_traces, CorrectMotionRecording


repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
remote_path = 'mearec/mearec_test_10s.h5'


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"




def test_correct_motion_on_peaks():
    pass

def test_correct_motion_on_traces():
    pass

def test_CorrectMotionRecording():
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    rec = MEArecRecordingExtractor(local_path)

    # make a fake motion vector
    duration = rec.get_total_duration()
    locs = rec.get_channel_locations()
    temporal_bins = np.arange(0.5, duration-0.49, 0.5)
    spatial_bins = np.arange(locs[:, 1].min(), locs[:, 1].max(), 100)
    motion =  np.zeros((temporal_bins.size, spatial_bins.size))
    motion[:, :] = np.linspace(-30, 30, temporal_bins.size)[:, None]

    rec2 = CorrectMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode='force_extrapolate')
    assert rec2.channel_ids.size == 32

    rec2 = CorrectMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode='force_zeros')
    assert rec2.channel_ids.size == 32

    rec2 = CorrectMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode='remove_channels')
    assert rec2.channel_ids.size == 26
    for ch_id in ('1', '11', '12', '21', '22', '23'):
        assert  ch_id not in rec2.channel_ids

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000)
    assert traces.shape == (30000, 26)

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000, channel_ids=['3', '4'])
    assert traces.shape == (30000, 2)


    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # fig, ax = plt.subplots()
    # sw.plot_probe_map(rec, with_channel_ids=True, ax=ax)
    # fig, ax = plt.subplots()
    # sw.plot_probe_map(rec2, with_channel_ids=True, ax=ax)
    # plt.show()



    






if __name__ == '__main__':
    # test_correct_motion_on_peaks()
    # test_correct_motion_on_traces()
    test_CorrectMotionRecording()
