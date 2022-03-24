from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor
import spikeinterface.sortingcomponents as scp
from spikeinterface import append_recordings

repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
remote_path = 'mearec/mearec_test_10s.h5'
mea_recording_path = download_dataset(repo=repo, remote_path=remote_path)
mea_recording = MEArecRecordingExtractor(mea_recording_path)

# for testing on data with multiple segments
frames_num = mea_recording.get_num_frames()
rec_part1 = mea_recording.frame_slice(0, frames_num / 2)
rec_part2 = mea_recording.frame_slice(frames_num / 2 + 1, frames_num)
segmented_mea_recording = append_recordings([rec_part1, rec_part2])

# will overwrite the default parameters but is optional
detection_param = {
    'out_file_name': "HS2_detected",
    'to_localize': False
}

hs2_peaks = scp.hs2_detect_spikes(recording=segmented_mea_recording, parameters=detection_param)

print(hs2_peaks)