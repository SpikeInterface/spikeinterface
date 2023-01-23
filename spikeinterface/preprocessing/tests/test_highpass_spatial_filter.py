import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import pytest
import numpy as np
from copy import deepcopy

from spikeinterface.core.testing_tools import generate_recording
import spikeinterface.widgets as sw

try:
    import spikeglx
    import neurodsp.voltage as voltage
    HAVE_IBL_NPIX = True
except ImportError:
    HAVE_IBL_NPIX = False

DEBUG = True
if DEBUG:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lagc", ["ibl", None, 1, 150])
def test_highpass_spatial_filter_real_data(lagc):
    """
    Test highpass spatial filter IBL vs. SI implimentations. Download
    a real dataset from SI repo, run both pipelines and check results.

    Requires a bit of exchanging argument types for differences in inputs
    between IBL and SI.

    Cycle through the lagc (automatic gain control) arguments (default
    is IBL default params, False and None are no lagc. Do this here rather
    than in larger test below to avoid unecessary tests, of every
    combination with False and None.

    Argument changes:
        - lagc is changed to a dictionary, so that sampling_interval
          can be easily passed to IBL helper function, and so that
          window_length can be specified in seconds.

        - ntr_pad can be None for SI, lagc can be "default", None or False.

    use DEBUG = true to visualise.

    """
    options = dict(lagc=lagc, ntr_pad=25, ntr_tap=50, butter_kwargs=None)
    print(options)

    ibl_data, si_recording = get_ibl_si_data()

    si_filtered, __ = run_si_highpass_filter(si_recording,
                                             **options)

    ibl_filtered = run_ibl_highpass_filter(ibl_data.copy(),
                                           **options)

    if DEBUG:
        fig, axs = plt.subplots(ncols=4)
        axs[0].imshow(si_recording.get_traces(return_scaled=True))
        axs[0].set_title("SI Raw")
        axs[1].imshow(ibl_data.T)
        axs[1].set_title("IBL Raw")
        axs[2].imshow(si_filtered)
        axs[2].set_title("SI Filtered ")
        axs[3].imshow(ibl_filtered)
        axs[3].set_title("IBL Filtered")

    assert np.allclose(si_filtered,
                       ibl_filtered*1e6,
                       atol=1e-01,
                       rtol=0)  # the differences are entired due to scaling on data load.

@pytest.mark.skip("Something wrong with the generation of the data")
@pytest.mark.parametrize("num_channels", [32, 64, 384])
def test_add_trend_and_check_deleted(num_channels):
    """
    Nice example to see what the funcntion is doing sanity check it performs
    as expected. Create a dataset (data_no_trend) with high spatial frequency
    in both x and y. Create a trend with low spatial frequency in y only. Add
    them together, filter and check only the low-frequency trend is y is removed.

    use DEBUG = true to visualise.

    the frequency response of the butter filter at the tested frequencies is:
    Wn 0.2 = -0.06 dB, Wn 0.01 is = -60dB
    """
    recording = generate_recording(num_channels=num_channels,
                                   durations=[0.5])
    trend, data_no_trend = make_trend_and_data_no_trend(recording,
                                                        trend_nyq=0.01,
                                                        data_no_trend_nyq=0.2)

    data = data_no_trend + trend
    recording._recording_segments[0]._traces = data

    __, si_highpass_spatial_filter = run_si_highpass_filter(recording,
                                                            ntr_pad=10, ntr_tap=10, lagc=None,
                                                            butter_kwargs=dict(N=3, Wn=0.1),
                                                            get_traces=False)
    si_filtered = si_highpass_spatial_filter.get_traces()

    if DEBUG:
        fig, ax = plt.subplots(ncols=1)
        ax.plot(trend[num_channels // 2], label="trend")
        ax.plot(data_no_trend[num_channels // 2], label="data no trend")
        ax.plot(data[num_channels // 2], label="trend + data_no_trend")
        ax.plot(si_filtered[num_channels // 2], label="filtered")
        ax.legend()

        # axs[0].imshow(trend)
        # axs[0].set_title("trend")
        # axs[1].imshow(data_no_trend)
        # axs[1].set_title("data_no_trend")
        # axs[1].imshow(data)
        # axs[2].set_title("trend + data_no_trend")
        # axs[3].imshow(si_filtered)
        # axs[3].set_title("filtered")

    corr_filtered_data_with_trend = np.corrcoef(si_filtered.ravel(), trend.ravel())
    assert corr_filtered_data_with_trend[0,1] < 0.01

    corr_filtered_data_with_original = np.corrcoef(si_filtered.ravel(), data_no_trend.ravel())
    assert corr_filtered_data_with_original[0,1] > 0.99


@pytest.mark.parametrize("ntr_pad", [None, 0, 31])
@pytest.mark.parametrize("ntr_tap", [None, 5])
@pytest.mark.parametrize("lagc", ["ibl", None, 1232])
@pytest.mark.parametrize("butter_kwargs", [None,
                                           {'N': 5, 'Wn': 0.12}])
@pytest.mark.parametrize("num_channels", [32, 64])
def test_highpass_spatial_filter_synthetic_data(num_channels, ntr_pad, ntr_tap, lagc, butter_kwargs):
    """
    Generate a short recording, run it through SI and IBL version, check outputs match. Used to
    check many combinations of possible inputs.
    """
    num_segments = 2
    options = dict(lagc=lagc, ntr_pad=ntr_pad, ntr_tap=ntr_tap, butter_kwargs=butter_kwargs)

    durations = [0.5, 0.8]
    si_recording = generate_recording(num_channels=num_channels,
                                      durations=durations)

    __, si_highpass_spatial_filter = run_si_highpass_filter(si_recording,
                                                            get_traces=False,
                                                            **options)
    for seg in range(num_segments):
        si_filtered = si_highpass_spatial_filter.get_traces(segment_index=seg)

        ibl_filtered = run_ibl_highpass_filter(ibl_data=si_recording.get_traces(segment_index=seg).T,  **options)

        assert np.allclose(si_filtered, ibl_filtered, atol=1e-06, rtol=0)


# ----------------------------------------------------------------------------------------------------------------------
# Test Utils
# ----------------------------------------------------------------------------------------------------------------------

def get_ibl_si_data():
    """
    Set fixture to session to ensure origional data is not changed.
    """
    local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
    ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True)
    ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel

    si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
    si_recording = spre.scale(si_recording, dtype="float32")

    return ibl_data, si_recording


def make_trend_and_data_no_trend(recording, trend_nyq, data_no_trend_nyq):
    """
    Make a 2D image, with a DC trend across 'channels' (y axis), the
    other a high-frequency change in x and y, that should not be much
    affected by removal of slow-frequency trend.
    """
    num_channels = recording.get_num_channels()
    magnitude = 0.5 * num_channels * recording.get_num_samples()

    y_dc_trend = np.zeros((recording.get_num_samples(), num_channels))
    nyq_samples = np.ceil((num_channels / 2) * trend_nyq).astype("int")
    y_dc_trend[0, nyq_samples] = magnitude
    y_dc_trend[0, -nyq_samples] = magnitude

    high_freq_x_and_y = np.zeros((recording.get_num_samples(), num_channels))
    nyq_samples = np.floor((num_channels / 2) * data_no_trend_nyq).astype("int")
    high_freq_x_and_y[nyq_samples, nyq_samples] = magnitude
    high_freq_x_and_y[-nyq_samples, -nyq_samples] = magnitude

    trend = np.abs(np.fft.ifft2(y_dc_trend))
    data_no_trend = np.abs(np.fft.ifft2(high_freq_x_and_y))

    return trend, data_no_trend


def process_args_for_si(si_recording, lagc):
    """"""
    if isinstance(lagc, int):
        ts = 1 / si_recording.get_sampling_frequency()
        window_s = lagc * ts
        si_lagc = {"window_length_s": window_s,
                   "sampling_interval": ts}
    else:
        si_lagc = lagc

    return si_lagc


def process_args_for_ibl(butter_kwargs, ntr_pad, lagc):
    """"""
    if butter_kwargs is None:
        butter_kwargs_ = {'N': 3, 'Wn': 0.01}
    else:
        butter_kwargs_ = deepcopy(butter_kwargs)
    if "btype" not in butter_kwargs_:
        butter_kwargs_['btype'] = 'highpass'

    if ntr_pad is None:
        ntr_pad = 0
    if lagc == "ibl":
        lagc = 300
    if lagc in [None, False]:
        lagc = 0

    return butter_kwargs_, ntr_pad, lagc


def run_si_highpass_filter(si_recording, ntr_pad, ntr_tap, lagc, butter_kwargs, get_traces=True):
    """"""
    si_lagc = process_args_for_si(si_recording, lagc)

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


def run_ibl_highpass_filter(ibl_data, ntr_pad, ntr_tap, lagc, butter_kwargs):

    butter_kwargs, ntr_pad, lagc = process_args_for_ibl(butter_kwargs,
                                                        ntr_pad,
                                                        lagc)

    ibl_filtered = voltage.kfilt(ibl_data, None, ntr_pad, ntr_tap, lagc, butter_kwargs).T

    return ibl_filtered


if __name__ == '__main__':
    test_highpass_spatial_filter_real_data(lagc="ibl")  # TODO: this will fail with fixtures...
    test_add_trend_and_check_deleted(num_channels=384)
    test_highpass_spatial_filter_synthetic_data()