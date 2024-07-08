import pytest
import numpy as np

from spikeinterface import NumpyRecording, get_random_data_chunks
from probeinterface import generate_linear_probe

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import detect_bad_channels, highpass_filter

try:
    # WARNING : this is not this package https://pypi.org/project/neurodsp/
    # BUT this one https://github.com/int-brain-lab/ibl-neuropixel
    # pip install ibl-neuropixel
    import neurodsp.voltage

    HAVE_NPIX = True
except:  # Catch relevant exception
    HAVE_NPIX = False


def test_detect_bad_channels_std_mad():
    num_channels = 4
    sampling_frequency = 30000.0
    durations = [10.325, 3.5]

    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype("float32")
        # one channel have big noise
        traces[:, 1] *= 10
        times = np.arange(num_timepoints[i]) / sampling_frequency
        traces += np.sin(2 * np.pi * 50 * times)[:, None]
        traces_list.append(traces)
    rec = NumpyRecording(traces_list, sampling_frequency)

    probe = generate_linear_probe(num_elec=num_channels)
    probe.set_device_channel_indices(np.arange(num_channels))
    rec.set_probe(probe, in_place=True)

    bad_channels_std, bad_labels_std = detect_bad_channels(rec, method="std")
    bad_channels_mad, bad_labels_mad = detect_bad_channels(rec, method="std")
    np.testing.assert_array_equal(bad_channels_std, bad_channels_mad)
    rec2 = rec.remove_channels(bad_channels_std)

    # Check that the noisy channel is taken out
    assert np.array_equal(rec2.get_channel_ids(), [0, 2, 3]), "wrong channel detected."
    # Check that the number of segments is maintained after preprocessor
    assert np.array_equal(rec2.get_num_segments(), rec.get_num_segments()), "wrong numbber of segments."
    # Check that the size of the segments os maintained after preprocessor
    assert np.array_equal(
        *([r.get_num_frames(x) for x in range(rec.get_num_segments())] for r in [rec, rec2])
    ), "wrong lenght of resulting segments."
    # Check that locations are mantained
    assert np.array_equal(
        rec.get_channel_locations()[[0, 2, 3]], rec2.get_channel_locations()
    ), "wrong channels locations."


@pytest.mark.parametrize("outside_channels_location", ["bottom", "top", "both"])
def test_detect_bad_channels_extremes(outside_channels_location):
    num_channels = 64
    sampling_frequency = 30000.0
    durations = [20]
    num_out_channels = 10

    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype("float32")
        # extreme channels are "out"
        traces[:, :num_out_channels] *= 0.05
        traces[:, -num_out_channels:] *= 0.05
        traces_list.append(traces)

    rec = NumpyRecording(traces_list, sampling_frequency)
    rec.set_channel_gains(1)
    rec.set_channel_offsets(0)

    probe = generate_linear_probe(num_elec=num_channels)
    probe.set_device_channel_indices(np.arange(num_channels))
    rec.set_probe(probe, in_place=True)

    bad_channel_ids, bad_labels = detect_bad_channels(
        rec, method="coherence+psd", outside_channels_location=outside_channels_location
    )
    if outside_channels_location == "top":
        assert np.array_equal(bad_channel_ids, rec.channel_ids[-num_out_channels:])
    elif outside_channels_location == "bottom":
        assert np.array_equal(bad_channel_ids, rec.channel_ids[:num_out_channels])
    elif outside_channels_location == "both":
        assert np.array_equal(
            bad_channel_ids, np.concatenate((rec.channel_ids[:num_out_channels], rec.channel_ids[-num_out_channels:]))
        )


@pytest.mark.skipif(not HAVE_NPIX, reason="ibl-neuropixel is not installed")
@pytest.mark.parametrize("num_channels", [32, 64, 384])
def test_detect_bad_channels_ibl(num_channels):
    """
    Cannot test against DL datasets because they are too short
    and need to control the PSD scaling. Here generate a dataset
    by using SI generate_recording() and adding dead and noisy channels
    to be detected. Use the PSD in between the generated high
    and low as the threshold.

    Test the full SI pipeline using chunks, and taking the mode,
    against the original IBL function using all data.

    IBL scale the psd to 1e6 but SI does not need to do this,
    however for testing it is necssary. So before calling the IBL function
    we need to rescale the traces to Volts.
    """
    # download_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
    # recording = se.read_spikeglx(download_path, stream_id="imec0.ap")
    recording = generate_recording(num_channels=num_channels, durations=[1])
    recording = highpass_filter(recording)
    # add "scale" to uV (test recording is alreayd in uV)
    recording.set_channel_gains(1)
    recording.set_channel_offsets(0)

    # Generate random channels to be dead / noisy
    is_bad = np.random.choice(
        np.arange(num_channels - 3), size=np.random.randint(5, int(num_channels * 0.25)), replace=False
    )
    is_noisy, is_dead = np.array_split(is_bad, 2)
    not_noisy = np.delete(np.arange(num_channels), is_noisy)

    # Add bad channels to data and test
    psd_cutoff = add_noisy_and_dead_channels(recording, is_dead, is_noisy, not_noisy)

    # Detect in SI
    bad_channel_ids, bad_channel_labels_si = detect_bad_channels(
        recording,
        method="coherence+psd",
        psd_hf_threshold=psd_cutoff,
        dead_channel_threshold=-0.5,
        noisy_channel_threshold=1,
        outside_channel_threshold=-0.75,
        seed=0,
    )

    # Detect in IBL (make sure we use the exact same chunks)
    random_chunk_kwargs = dict(
        num_chunks_per_segment=10,
        chunk_size=int(0.3 * recording.sampling_frequency),
        seed=0,
        concatenated=False,
        return_scaled=True,
    )

    random_data = get_random_data_chunks(recording, **random_chunk_kwargs)
    channel_flags_ibl = np.zeros((recording.get_num_channels(), recording.get_num_segments() * 10), dtype=int)
    for i, random_chunk in enumerate(random_data):
        traces_uV = random_chunk.T
        traces_V = traces_uV * 1e-6
        channel_flags, _ = neurodsp.voltage.detect_bad_channels(
            traces_V,
            recording.get_sampling_frequency(),
            psd_hf_threshold=psd_cutoff,
        )
        channel_flags_ibl[:, i] = channel_flags

    # Take the mode of the chunk estimates as final result. Convert to binary good / bad channel output.
    import scipy.stats

    bad_channel_labels_ibl, _ = scipy.stats.mode(channel_flags_ibl, axis=1, keepdims=False)

    # Compare
    channels_labeled_as_good = bad_channel_labels_si == "good"
    expected_channels_labeled_as_good = bad_channel_labels_ibl == 0
    assert np.array_equal(channels_labeled_as_good, expected_channels_labeled_as_good)

    channels_labeled_as_dead = bad_channel_labels_si == "dead"
    expected_channels_labeled_as_good = bad_channel_labels_ibl == 1
    assert np.array_equal(channels_labeled_as_dead, expected_channels_labeled_as_good)

    channels_labeled_as_noisy = bad_channel_labels_si == "noise"
    expected_channels_labeled_as_good = bad_channel_labels_ibl == 2
    assert np.array_equal(channels_labeled_as_noisy, expected_channels_labeled_as_good)

    assert np.array_equal(recording.ids_to_indices(bad_channel_ids), np.where(bad_channel_labels_ibl != 0)[0])

    # Test on randomly sorted channels
    recording_scrambled = recording.channel_slice(
        np.random.choice(recording.channel_ids, len(recording.channel_ids), replace=False)
    )
    bad_channel_ids_scrambled, bad_channel_label_scrambled = detect_bad_channels(
        recording_scrambled,
        method="coherence+psd",
        psd_hf_threshold=psd_cutoff,
        dead_channel_threshold=-0.5,
        noisy_channel_threshold=1,
        outside_channel_threshold=-0.75,
        seed=0,
    )
    assert all(bad_channel in bad_channel_ids for bad_channel in bad_channel_ids_scrambled)


def add_noisy_and_dead_channels(recording, is_dead, is_noisy, not_noisy):
    """ """
    psd_cutoff = reduce_high_freq_power_in_non_noisy_channels(recording, is_noisy, not_noisy)
    recording = add_dead_channels(recording, is_dead)
    # Note this will reduce the PSD for these channels but
    # as noisy have higher freq > 80% nyqist this does not matter

    return psd_cutoff


def reduce_high_freq_power_in_non_noisy_channels(recording, is_noisy, not_noisy):
    """
    Reduce power in >80% Nyquist for all channels except noisy channels to 20% of original.
    Return the psd_cutoff in uV^2/Hz that separates the good at noisy channels.
    """
    import scipy.signal

    for iseg, __ in enumerate(recording._recording_segments):
        data = recording.get_traces(iseg).T
        num_samples = recording.get_num_samples(iseg)

        step_80_percent_nyq = np.ones(num_samples) * 0.1
        step_80_percent_nyq[int(num_samples * 0.2) : int(num_samples * 0.8)] = 1

        # fft and multiply the shifted freqeuncies by 0.2 at > 80% nyquist, then unshift and ifft
        D = np.fft.fftshift(np.fft.fft(data[not_noisy])) * step_80_percent_nyq
        data[not_noisy] = np.fft.ifft(np.fft.ifftshift(D))

    # calculate the psd_cutoff (which separates noisy and non-noisy) ad-hoc from the last segment
    fscale, psd = scipy.signal.welch(data, fs=recording.get_sampling_frequency())
    psd_cutoff = np.mean([np.mean(psd[not_noisy, -50:]), np.mean(psd[is_noisy, -50:])])
    return psd_cutoff


def add_dead_channels(recording, is_dead):
    """
    Add 'dead' channels (low-amplitude (10% of original) white noise).
    """
    for segment_index in range(recording.get_num_segments()):
        data = recording.get_traces(segment_index)

        std = np.mean(np.std(data, axis=1))
        mean = np.mean(data)
        data[:, is_dead] = np.random.normal(
            mean, std * 0.1, size=(is_dead.size, recording.get_num_samples(segment_index))
        ).T
        recording._recording_segments[segment_index]._traces = data


if __name__ == "__main__":
    # test_detect_bad_channels_std_mad()
    test_detect_bad_channels_ibl(num_channels=32)
    test_detect_bad_channels_ibl(num_channels=64)
    test_detect_bad_channels_ibl(num_channels=384)
    # test_detect_bad_channels_extremes("top")
    # test_detect_bad_channels_extremes("bottom")
    # test_detect_bad_channels_extremes("both")
