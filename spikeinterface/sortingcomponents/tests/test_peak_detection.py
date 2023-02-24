import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass
from spikeinterface.sortingcomponents.features_from_peaks import PeakToPeakFeature

try:
    import pyopencl
    HAVE_PYOPENCL = True
except:
    HAVE_PYOPENCL = False

try:
    import torch
    HAVE_TORCH = True
except:
    HAVE_TORCH = False

def test_detect_peaks():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True, verbose=True)
    # by_channel
    by_channel_str = f"By channel:\n"
    peaks_by_channel_np = detect_peaks(recording, method='by_channel',
                                       peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                       **job_kwargs)
    by_channel_str += f"- numpy - {len(peaks_by_channel_np)}\n"

    if HAVE_TORCH:    
        peaks_by_channel_torch = detect_peaks(recording, method='by_channel_torch',
                                              peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                              **job_kwargs)
        # due to the different implementations, we allow a small tolerance
        assert np.isclose(np.array(len(peaks_by_channel_np)), np.array(len(peaks_by_channel_torch)),
                          rtol=0.1)
        by_channel_str += f"- torch - {len(peaks_by_channel_torch)}\n"
    print(by_channel_str)

    # locally_exclusive
    locally_exclusive_str = f"Locally exclusive:\n"
    peaks_local_numba = detect_peaks(recording, method='locally_exclusive',
                                     peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                     **job_kwargs)
    assert len(peaks_by_channel_np) > len(peaks_local_numba)

    locally_exclusive_str += f"- numba - {len(peaks_local_numba)}\n"

    if HAVE_TORCH:
        peaks_local_torch = detect_peaks(recording, method='locally_exclusive_torch',
                                         peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                         **job_kwargs)
        assert len(peaks_by_channel_torch) > len(peaks_local_torch)
        # due to the different implementations, we allow a small tolerance
        assert np.isclose(np.array(len(peaks_local_numba)), np.array(len(peaks_local_torch)),
                          rtol=0.1)
        locally_exclusive_str += f"- torch - {len(peaks_local_torch)}\n"

    
    # locally_exclusive + opencl
    if HAVE_PYOPENCL:
        peaks_local_cl = detect_peaks(recording, method='locally_exclusive_cl',
                                      peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                      **job_kwargs)
        locally_exclusive_str += f"- opencl - {len(peaks_local_cl)}\n"
        # in this case implementations are exactly the same
        assert len(peaks_local_numba) == len(peaks_local_cl)
    print(locally_exclusive_str)

    # locally_exclusive + pipeline steps LocalizeCenterOfMass + PeakToPeakFeature
    print("With peak pipeline")
    extract_dense_waveforms = ExtractDenseWaveforms(recording, ms_before=1., ms_after=1.,
                                                    return_output=False)

    pipeline_nodes = [
        extract_dense_waveforms,
        PeakToPeakFeature(recording,  all_channels=False, parents=[extract_dense_waveforms]),
        LocalizeCenterOfMass(recording, local_radius_um=50., parents=[extract_dense_waveforms]),
    ]
    peaks, ptp, peak_locations = detect_peaks(recording, method='locally_exclusive',
                                              peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                                              pipeline_nodes=pipeline_nodes, **job_kwargs)
    assert peaks.shape[0] == ptp.shape[0]
    assert peaks.shape[0] == peak_locations.shape[0]
    assert 'x' in peak_locations.dtype.fields

    if HAVE_TORCH:
        peaks_torch, ptp_torch, peak_locations_torch = detect_peaks(recording, method='locally_exclusive_torch',
                                                                    peak_sign='neg', detect_threshold=5,
                                                                    exclude_sweep_ms=0.1, pipeline_nodes=pipeline_nodes,
                                                                    **job_kwargs)
        assert peaks_torch.shape[0] == ptp_torch.shape[0]
        assert peaks_torch.shape[0] == peak_locations_torch.shape[0]
        assert 'x' in peak_locations_torch.dtype.fields

    if HAVE_PYOPENCL:
        peaks_cl, ptp_cl, peak_locations_cl = detect_peaks(recording, method='locally_exclusive_cl',
                                                           peak_sign='neg', detect_threshold=5,
                                                           exclude_sweep_ms=0.1, pipeline_nodes=pipeline_nodes,
                                                           **job_kwargs)
        assert peaks_cl.shape[0] == ptp_cl.shape[0]
        assert peaks_cl.shape[0] == peak_locations_cl.shape[0]
        assert 'x' in peak_locations_cl.dtype.fields
    

    # # DEBUG
    # sample_inds, chan_inds, amplitudes = peaks['sample_ind'], peaks['channel_ind'], peaks['amplitude']
    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # chan_offset = 500
    # traces = recording.get_traces()
    # traces += np.arange(traces.shape[1])[None, :] * chan_offset
    # fig, ax = plt.subplots()
    # ax.plot(traces, color='k')
    # ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color='r')
    # plt.show()

    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # from probeinterface.plotting import plot_probe
    # probe = recording.get_probe()
    # fig, ax = plt.subplots()
    # plot_probe(probe, ax=ax)
    # ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.5)
    # # MEArec is "yz" in 2D
    # import MEArec
    # recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
    # check_suffix=False,
    # load=['recordings', 'spiketrains', 'channel_positions'],
    # load_waveforms=False)
    # soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
    # for i, st in enumerate(recgen.spiketrains):
    #     soma_positions[i, :] = st.annotations['soma_position']
    # ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
    # plt.show()


if __name__ == '__main__':
    test_detect_peaks()
    
