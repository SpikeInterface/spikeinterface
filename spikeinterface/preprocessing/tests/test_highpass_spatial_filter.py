import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording

try:
    import spikeglx
    import neurodsp.voltage as voltage
    HAVE_IBL_NPIX = True
except ImportError:
    HAVE_IBL_NPIX = False

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()


@pytest.mark.skipif(not HAVE_IBL_NPIX, reason="Requires ibl-neuropixel install")
class TestHighPassFilter:

    @pytest.fixture(scope="function")
    def ibl_si_data(self):
        """
        Set fixture to session to ensure origional data is not changed.
        """
        local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
        ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True)
        ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel

        si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
        si_recording = spre.scale(si_recording, dtype="float32")

        return [ibl_data, si_recording]

    @pytest.mark.parametrize("lagc", ["default", None, False, 1])
    def test_highpass_spatial_filter_real_data(self, ibl_si_data, lagc):
        """
        Test highpass spatial filter IBL vs. SI implimentations.

        Requires a bit of exchanging argument types for differences in inputs
        between IBL and SI.

        Argument changes:
            - lagc is changed to a dictionary, so that sampling_interval
              can be easily passed to IBL helper function, and so that
              window_length can be specified in seconds.

            - ntr_pad can be None for SI, lagc can be "default", None or False.

        """
        options = dict(lagc=lagc, ntr_pad=25, ntr_tap=50, butter_kwargs=None)

        ibl_data, si_recording = ibl_si_data

        si_filtered, __ = self.run_si_highpass_filter(si_recording,
                                                  **options)

        ibl_filtered = self.run_ibl_highpass_filter(ibl_data,
                                                    **options)

        if DEBUG:
            for bad_idx in [0, 1, 2]:
                fig, ax = plt.subplots()
                ax.plot(si_filtered[:, bad_idx], label="SI")
                ax.plot(ibl_filtered[:, bad_idx] * 1e6, label="IBL")
                ax.set_title(f"bad channel {bad_idx}")
                ax.legend()

        assert np.allclose(si_filtered,
                           ibl_filtered*1e6,
                           atol=1e-01,
                           rtol=0)  # the differences are entired due to scaling on data load. If passing SI input to
                                    # this function results are the same 1e-08


    @pytest.mark.parametrize("ntr_pad", [None, 0, 10, 31])
    @pytest.mark.parametrize("ntr_tap", [None, 5, 31])
    @pytest.mark.parametrize("lagc", ["default", None, 125, 1232])
    @pytest.mark.parametrize("butter_kwargs", [None,
                                               {'N': 3, 'Wn': 0.05, 'btype': 'highpass'},
                                               {'N': 5, 'Wn': 0.12, 'btype': 'lowpass'}])
    @pytest.mark.parametrize("num_channels", [32, 64, 384])
    def test_highpass_spatial_filter_synthetic_data(self, num_channels, ntr_pad, ntr_tap, lagc, butter_kwargs):
        """
        """
        num_segments = 2
        options = dict(lagc=lagc, ntr_pad=ntr_pad, ntr_tap=ntr_tap, butter_kwargs=butter_kwargs)

        si_recording = self.get_test_recording(num_channels, num_segments)

        __, si_highpass_spatial_filter = self.run_si_highpass_filter(si_recording,
                                                                     get_traces=False,
                                                                     **options)
        for seg in range(num_segments):
            

            si_filtered = si_highpass_spatial_filter.get_traces(segment_index=seg)
   
            ibl_filtered = self.run_ibl_highpass_filter(ibl_data=si_recording.get_traces(segment_index=seg).T,  **options)

            assert np.allclose(si_filtered, ibl_filtered, atol=1e-06, rtol=0)

    def add_trend_and_check_deleted(self):

        si_recording = self.get_test_recording(num_channels, num_segments=1)

        # TODO: make a sanity check test, add spatially varing trendlind Wn 0.01
        #       determine units for 0.1 Wn in the spatial domain
        """
            
        1) the fft convolution very similar, for 
        n = 1000 on a 1035 length signal they are
        slightly different but in real life will never 
        have such a case. also only slightly different. or when wl is 1-2 samples
        
        in get_chunks:
        
        taper_ibl = fcn_cosine([0, margin])(np.arange(margin))

        """


    def process_args_for_si(self, si_recording, lagc):
        """"""
        if type(lagc) == int:
            ts = 1 / si_recording.get_sampling_frequency()
            window_s = lagc * ts
            si_lagc = {"window_length_s": window_s,
                       "sampling_interval": ts}
        else:
            si_lagc = lagc

        return si_lagc

    def process_args_for_ibl(self, butter_kwargs, ntr_pad, lagc):
        """"""
        if butter_kwargs is None:
            butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

        if ntr_pad is None:
            ntr_pad = 0
        if lagc == "default":
            lagc = 300
        if lagc in [None, False]:
            lagc = 0

        return butter_kwargs, ntr_pad, lagc

    def get_test_recording(self, num_channels=32, num_segments=2):
        """
        1500 and 2100 samples
        """
        recording = generate_recording(num_channels=num_channels,
                                       durations=[0.05, 0.07])

        return recording

    def run_si_highpass_filter(self, si_recording, ntr_pad, ntr_tap, lagc, butter_kwargs, get_traces=True):
        """"""
        si_lagc = self.process_args_for_si(si_recording, lagc)

        si_highpass_spatial_filter = spre.highpass_spatial_filter(si_recording,
                                                                  n_channel_pad=ntr_pad,
                                                                  n_channel_taper=ntr_tap,
                                                                  agc_options=si_lagc,
                                                                  butter_kwargs=butter_kwargs)

        if get_traces:
            si_filtered = si_highpass_spatial_filter.get_traces(return_scaled=True)
        else:
            si_filtered = False

        return si_filtered, si_highpass_spatial_filter

    def run_ibl_highpass_filter(self, ibl_data, ntr_pad, ntr_tap, lagc, butter_kwargs):

        butter_kwargs, ntr_pad, lagc = self.process_args_for_ibl(butter_kwargs,
                                                                 ntr_pad,
                                                                 lagc)

        ibl_filtered = voltage.kfilt(ibl_data, None, ntr_pad, ntr_tap, lagc, butter_kwargs).T

        return ibl_filtered
