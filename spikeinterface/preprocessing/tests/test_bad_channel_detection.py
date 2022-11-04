# assert for uV SI!
# TODO: is the assmption that large channel number == outside of brain true across all probes?
# TODO: makes the assumption that 5000 Hz is LFP band - is this valid for other probes?
# TODO: assumes segments are all the same (pools all when getting random chunks - is this valid?) 
import copy
import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from pathlib import Path
import numpy as np
import pytest
import scipy.signal
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.preprocessing.remove_bad_channels import detect_bad_channels_ibl

try:
    import spikeglx
    import neurodsp.voltage as voltage
except:  # Catch relevant exception
    raise ImportError("Requires ibl-neuropixel dev install (pip install -e .) from inside cloned repo."
                      "https://github.com/int-brain-lab/ibl-neuropixel")


class TestBadChannelDetection():

    @pytest.fixture(scope="class")
    def recording(self):
        """
        Set fixture to class to ensure origional data is not changed.
        """
        download_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
        recording = se.read_spikeglx(download_path, stream_id="imec0.ap")

        yield recording

    @pytest.mark.parametrize("set_num_channels", [32, 64, 384])
    def test_download_data_against_ibl_neuropixel(self, recording, set_num_channels):
        """
        Cannot test against DL datasets because they are too short
        and need to control the PSD scaling. Here generate a dataset
        by using SI get_test_recording() and adding dead and noisy channels
        to be detected. Use the PSD in between the generated high
        and low as the threshold.

        Test the full SI pipeline using chunks, and taking the mode,
        against the original IBL function using all data.

        IBL scale the psd to 1e6 but SI does not need to do this,
        however for testing it is necssary. Thus (TODO) there is a
        hidden flag, if the key "scale_for_testing" is included in
        the random_chunk_kwargs then the psd in SI bad_channel_detection
        will be scaled by 1e6. This also requires scaling the PSD
        """
        num_channels = set_num_channels
        __, recording = self.get_test_recording(num_channels)

        # Generate random channels to be dead / noisy
        is_bad = np.random.choice(np.arange(num_channels - 3),
                                  size=np.random.randint(5, int(num_channels * 0.25)),
                                  replace=False)

        is_noisy, is_dead = np.array_split(is_bad, 2)
        not_noisy = np.delete(np.arange(num_channels), is_noisy)

        # Add bad channels to data and test
        psd_cutoff = self.add_noisy_and_dead_channels_to_all_segments(recording,
                                                                      is_dead,
                                                                      is_noisy,
                                                                      not_noisy,
                                                                      scale_psd=1e6)

        # Detect in SI
        random_kwargs = self.get_chunk_kwargs(recording, use_method_defaults=True, add_scale_for_testing=True)

        bad_inds, bad_channel_ids, channel_flags = spre.detect_bad_channels(recording,
                                                                            psd_hf_threshold=psd_cutoff,
                                                                            random_chunk_kwargs=random_kwargs)

        # Detect in IBL
        channel_flags_ibl, __ = voltage.detect_bad_channels(recording.get_traces().T,
                                                            recording.get_sampling_frequency(),
                                                            psd_hf_threshold=psd_cutoff)

        # Compare
        assert np.array_equal(np.where(channel_flags == 0), np.where(channel_flags_ibl.ravel() == 0))
        assert np.array_equal(np.where(channel_flags == 1), np.where(channel_flags_ibl.ravel() == 1))
        assert np.array_equal(np.where(channel_flags == 2), np.where(channel_flags_ibl.ravel() == 2))

        assert np.array_equal(bad_inds, np.where(channel_flags_ibl != 0)[0])


    def get_chunk_kwargs(self, recording, use_method_defaults=True, add_scale_for_testing=False):
        """
        Return kwargs for proper chunking behaviour for the test dataset. Either using
        the implementation defaults or returning 1 lage chunk per segment covering
        nearly all data points.
        """
        random_kwargs = {"return_scaled": False}

        if add_scale_for_testing:
            random_kwargs.update({"scale_for_testing": "always_on"})

        if not use_method_defaults:
            chunk_samples = recording.get_num_samples(0)-2
            random_kwargs.update({"num_chunks_per_segment": 1, "chunk_size": chunk_samples})

        return random_kwargs

    def add_noisy_and_dead_channels_to_all_segments(self, recording, is_dead, is_noisy, not_noisy, scale_psd):
        """
        """
        psd_cutoff = self.reduce_high_freq_power_in_non_noisy_channels_all_segments(recording,
                                                                                   is_noisy,
                                                                                   not_noisy,
                                                                                   scale_psd)

        self.add_dead_channels_to_all_segments(recording, is_dead)  # Note this will reduce the PSD for these channels but
                                                                    # as noisy have higher freq > 80% nyqist this does not matter

        return psd_cutoff

    def reduce_high_freq_power_in_non_noisy_channels_all_segments(self, recording, is_noisy, not_noisy, scale_psd):
        """
        Reduce power in >80% Nyquist for all channels except noisy channels to 20% of original.
        Return the psd_cutoff in uV^2/Hz that separates the good at noisy channels.
        """
        for iseg, __ in enumerate(recording._recording_segments):
            data = recording.get_traces(iseg).T
            num_samples = recording.get_num_samples(iseg)

            step_80_percent_nyq = np.ones(num_samples) * 0.1
            step_80_percent_nyq[int(num_samples * 0.2):int(num_samples * 0.8)] = 1

            # fft and multiply the shifted freqeuncies by 0.2 at > 80% nyquist, then unshift and ifft
            D = np.fft.fftshift(np.fft.fft(data[not_noisy])) * step_80_percent_nyq
            data[not_noisy] = np.fft.ifft(np.fft.ifftshift(D))

        # calculate the psd_cutoff (which separates noisy and non-noisy) ad-hoc from the last segment
        fscale, psd = scipy.signal.welch(data * scale_psd,
                                         fs=recording.get_sampling_frequency())
        psd_cutoff = np.mean([np.mean(psd[not_noisy, -50:]),
                              np.mean(psd[is_noisy, -50:])])
        return psd_cutoff

    def add_dead_channels_to_all_segments(self, recording, is_dead):
        """
        Add 'dead' channels (low-amplitude (10% of original) white noise).
        """
        for iseg, __ in enumerate(recording._recording_segments):
            data = recording.get_traces(iseg)

            std = np.mean(np.std(data, axis=1))
            mean = np.mean(data)
            data[:, is_dead] = np.random.normal(mean, std * 0.1,
                                                size=(is_dead.size, recording.get_num_samples(iseg))).T

    @staticmethod
    def get_test_recording(num_channels=32):

        recording = generate_recording(num_channels=num_channels,
                                       durations=[1])
        return num_channels, recording
