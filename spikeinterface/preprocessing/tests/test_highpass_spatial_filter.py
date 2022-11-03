import spikeglx
import neurodsp.voltage as voltage
import spikeinterface as si
import spikeinterface.preprocessing as sipp
import spikeinterface.extractors as se
import pytest
import numpy as np
import copy


class TestHighPassFilter:
    
    @pytest.fixture(scope="function")
    def ibl_si_data(self):
        """
        Set fixture to class to ensure origional data is not changed.
        """
        local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
        ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True)

        si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
        si_scaled_recording = sipp.ScaleRecording(si_recording,
                                                  gain=si_recording.get_property("gain_to_uV"),
                                                  offset=si_recording.get_property("offset_to_uV"))

        return [ibl_recording, si_scaled_recording]

    @pytest.mark.parametrize("select_channel_idx", [None, True])
    @pytest.mark.parametrize("ntr_pad", [None, 0, 10, 25, 50, 100])
    @pytest.mark.parametrize("ntr_tap", [None, 10, 25, 50, 100])
    @pytest.mark.parametrize("lagc", ["default", None, False, 1, 300, 600, 1000])
    @pytest.mark.parametrize("butter_kwargs", [None, {'N': 3, 'Wn': 0.05, 'btype': 'highpass'}, {'N': 5, 'Wn': 0.12, 'btype': 'lowpass'}])
    def test_highpass_spatial_filter_ibl_vs_si(self, ibl_si_data, ntr_pad, ntr_tap, lagc, butter_kwargs, select_channel_idx):
        """
        Test highpass spatial filter IBL vs. SI implimentations.

        Requires a bit of exchanging argument types for differences in inputs
        between IBL and SI.

        Argument changes:
            - lagc is changed to a dictionary, so that sampling_interval
              can be easily passed to IBL helper function, and so that
              window_length can be specified in seconds.

            - select_channel_idx: pass an array of channel indicies
              rather than bool array for consistency with other SI functions.
              IBL function is resursive and kfilt default settings
              are: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}. So to
              test with select_channel_idx butter_kwargs are changed
              for SI too.
            - ntr_pad can be None for SI, lagc can be "default", None or False.

        """
        ibl_recording, si_scaled_recording = ibl_si_data
        ibl_data = ibl_recording.read()[0].T[:-1, :] * 1e6

        if select_channel_idx:
            select_channel_idx, butter_kwargs = self.process_select_channel_idx()

        # Run SI highpass spatial filter

        si_lagc = self.process_args_for_si(si_scaled_recording, lagc)

        si_highpass_spatial_filter = sipp.highpass_spatial_filter(si_scaled_recording,
                                                                  n_channel_pad=ntr_pad,
                                                                  n_channel_taper=ntr_tap,
                                                                  agc_options=si_lagc,
                                                                  butter_kwargs=butter_kwargs,
                                                                  select_channel_idx=select_channel_idx)
        si_filtered = si_highpass_spatial_filter.get_traces()

        # Run IBL highpass spatial filter

        butter_kwargs, collection, ntr_pad, lagc = self.process_args_for_ibl(si_scaled_recording,
                                                                             butter_kwargs,
                                                                             select_channel_idx,
                                                                             ntr_pad,
                                                                             lagc)

        ibl_filtered = voltage.kfilt(ibl_data, collection, ntr_pad, ntr_tap, lagc, butter_kwargs)

        assert np.allclose(si_filtered,
                           ibl_filtered,
                           atol=1e-02,
                           rtol=0)  # the differences are entired due to scaling on data load. If passing SI input to
                                    # this function results are the same 1e-08


    def process_select_channel_idx(self):
        """"""
        select_channel_idx = np.random.choice(374, 20, replace=False) + 10
        butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}

        return select_channel_idx, butter_kwargs

    def process_args_for_si(self, si_scaled_recording, lagc):
        """"""
        if type(lagc) == int:
            ts = 1 / si_scaled_recording.get_sampling_frequency()
            window_s = lagc * ts
            si_lagc = {"window_length_s": window_s,
                       "sampling_interval": ts}
        else:
            si_lagc = lagc

        return si_lagc

    def process_args_for_ibl(self, si_scaled_recording, butter_kwargs, select_channel_idx, ntr_pad, lagc):
        """"""
        if butter_kwargs is None:
            butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

        if np.any(select_channel_idx):
            collection = np.zeros(si_scaled_recording.get_num_channels(), dtype=bool)
            collection[select_channel_idx] = True
        else:
            collection = None

        if ntr_pad is None:
            ntr_pad = 0
        if lagc == "default":
            lagc = 300
        if lagc in [None, False]:
            lagc = 0

        return butter_kwargs, collection, ntr_pad, lagc