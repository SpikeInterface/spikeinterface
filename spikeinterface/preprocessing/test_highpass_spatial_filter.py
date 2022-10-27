import spikeglx
import neurodsp.voltage as voltage
import spikeinterface as si
import spikeinterface.preprocessing as sipp
import spikeinterface.extractors as se
import pytest
import numpy as np
import copy

#
collection = None
ntr_pad = 50
ntr_tap = 50
lagc = 300
butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

class TestHighPassFilter:
    
    @pytest.fixture(scope="function")
    def ibl_si_data(self):

        local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
        ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True)

        si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
        si_scaled_recording = sipp.ScaleRecording(si_recording,
                                                  gain=si_recording.get_property("gain_to_uV"),
                                                  offset=si_recording.get_property("offset_to_uV"))

        return [ibl_recording, si_scaled_recording]

    @pytest.mark.parametrize("collection", [None])  # TODO: handle collection, [None, np.random.choice(374, 20, replace=False) + 10])
    @pytest.mark.parametrize("ntr_pad", [0, 10, 25, 50, 100])  # TODO: handle None
    @pytest.mark.parametrize("ntr_tap", [10, 25, 50, 100])     # TODO: handle None
    @pytest.mark.parametrize("lagc", [1, 300, 600, 1000])      # TODO: handle None, 0
    @pytest.mark.parametrize("butter_kwargs", [None, {'N': 3, 'Wn': 0.05, 'btype': 'highpass'}])
    def test_highpass_spatial_filter_ibl_vs_si(self, ibl_si_data, collection, ntr_pad, ntr_tap, lagc, butter_kwargs):

        ibl_recording, si_scaled_recording = ibl_si_data

        if lagc is not None:
            ts = 1 / si_scaled_recording.get_sampling_frequency()
            window_s = lagc * ts
            si_lagc = {"window_length_s": window_s,
                       "sampling_interval": ts}
        else:
            si_lagc = None

        si_highpass_spatial_filter = sipp.highpass_spatial_filter(si_scaled_recording,
                                                                  collection=collection,
                                                                  n_channel_pad=ntr_pad,
                                                                  n_channel_taper=ntr_tap,
                                                                  agc_options=si_lagc,
                                                                  butter_kwargs=butter_kwargs)
        si_filtered = si_highpass_spatial_filter.get_traces()


        if butter_kwargs is None:
            butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}
        
         data = copy.deepcopy(si_scaled_recording.get_traces().T)
   #     data = ibl_recording.read()[0].T[:-1, :] * 1e6
 
        ibl_filtered = voltage.kfilt(data, collection, ntr_pad, ntr_tap, lagc, butter_kwargs)

        assert np.allclose(si_filtered, ibl_filtered, 1e-0)
