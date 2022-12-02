import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import pytest
import numpy as np
from spikeinterface.core.testing_tools import generate_recording
import spikeinterface.widgets as sw

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


@pytest.mark.parametrize("lagc", ["default", None, False, 1])
def test_highpass_spatial_filter_real_data(ibl_si_data, lagc):
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

    ibl_data, si_recording = get_ibl_si_data()

    si_filtered, __ = run_si_highpass_filter(si_recording,
                                              **options)

    ibl_filtered = run_ibl_highpass_filter(ibl_data,
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
@pytest.mark.parametrize("lagc", ["default", None, 150, 1232])
@pytest.mark.parametrize("butter_kwargs", [None,
                                           {'N': 3, 'Wn': 0.05, 'btype': 'highpass'},
                                           {'N': 5, 'Wn': 0.12, 'btype': 'lowpass'}])
@pytest.mark.parametrize("num_channels", [32, 64, 384])
def test_highpass_spatial_filter_synthetic_data(num_channels, ntr_pad, ntr_tap, lagc, butter_kwargs):
    """
    """
    num_segments = 2
    options = dict(lagc=lagc, ntr_pad=ntr_pad, ntr_tap=ntr_tap, butter_kwargs=butter_kwargs)

    si_recording = get_test_recording(num_channels, num_segments)

    __, si_highpass_spatial_filter = run_si_highpass_filter(si_recording,
                                                                 get_traces=False,
                                                                 **options)
    for seg in range(num_segments):


        si_filtered = si_highpass_spatial_filter.get_traces(segment_index=seg)

        ibl_filtered = run_ibl_highpass_filter(ibl_data=si_recording.get_traces(segment_index=seg).T,  **options)

        assert np.allclose(si_filtered, ibl_filtered, atol=1e-06, rtol=0)

@pytest.mark.parametrize("num_channels", [32, 64, 384])
def test_add_trend_and_check_deleted(num_channels):

    recording = get_test_recording(num_channels, num_segments=1)

    import matplotlib.pyplot as plt

#    channel_trend = get_trend_across_channels(recording.get_num_samples(),
 #                                                  num_channels,
  #                                            fill_value=(np.max(recording.get_traces()) * recording.get_num_samples()*num_channels)/2,
   #                                           nyq_percents_x=[0.01, 0.02],
    #                                          nyq_percents_y=[0.2])

    k_space = np.zeros((recording.get_num_samples(), num_channels))
    fill_value = 1 * num_channels * recording.get_num_samples()

    nyq_samples = np.floor((num_channels / 2) * 0.005).astype("int")
    k_space[0, nyq_samples] = fill_value
    k_space[0, -nyq_samples] = fill_value

    nyq_samples = np.floor((num_channels / 2) * 0.1).astype("int")
    k_space[nyq_samples, nyq_samples] = fill_value
    k_space[nyq_samples, -nyq_samples] = fill_value

    channel_trend = np.abs(np.fft.ifft2(k_space))

    plt.imshow(channel_trend)
    plt.show()

 #   breakpoint()
    data = recording.get_traces()
   # channel_trend = np.sin(2 * np.pi * 2 * np.linspace(0, 1, num_channels))
    #channel_trend = np.tile(channel_trend[:, None], recording.get_num_samples()).T  # TODO, better way?
    #channel_trend -= np.mean(channel_trend)

    import scipy.signal

#    plt.imshow(data)
 #   plt.show()
  #  sos = scipy.signal.butter(N=3, Wn=0.05, btype="lowpass", output='sos')
   # data = scipy.signal.sosfiltfilt(sos, data, axis=1)
  #  breakpoint()
 #   new_data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, recording.get_num_samples())) + np.random.rand(recording.get_num_samples())
  #  new_data = np.tile(new_data[:, None], num_channels)

  #  new_data = np.linspace(-1, 1, num_channels)
   # new_data = np.tile(new_data[:, None], num_channels)
   # new_data = np.random.rand(recording.get_num_samples(), num_channels)#* 100000
    #new_data -= np.mean(new_data)

    recording._recording_segments[0]._traces = np.zeros(data.shape) #new_data
    data = recording.get_traces()
    data_no_trend = data.copy()

    plt.imshow(data)
    plt.show()

    data_with_trend = data + channel_trend
    recording._recording_segments[0]._traces = data_with_trend.copy()

    plt.imshow(recording.get_traces())
    plt.show()

    a_ = np.corrcoef(data_with_trend.ravel(), channel_trend.ravel())
    b_ = np.corrcoef(data_with_trend.ravel(), data_no_trend.ravel())
    breakpoint()

    __, si_highpass_spatial_filter = run_si_highpass_filter(recording, ntr_pad=10, ntr_tap=10, lagc=None, butter_kwargs=dict(N=3, Wn=0.1, btype="highpass"), get_traces=False)
    si_filtered = si_highpass_spatial_filter.get_traces()
#    si_filtered[np.abs(si_filtered) < 1e-1] = 0
    plt.imshow(si_filtered)
    plt.show()

    # could kspace againt and check there is power at once freq and not the other

    a_ = np.corrcoef(si_filtered.ravel(), channel_trend.ravel())
    b_ = np.corrcoef(si_filtered.ravel(), data_no_trend.ravel())
    breakpoint()

def get_trend_across_channels(num_samples, num_channels, nyq_percents_x, nyq_percents_y, fill_value=1):
    """ """
    k_space = np.zeros((num_samples, num_channels))

    for nyq_percent in nyq_percents_x:
        nyq_samples = np.floor((num_channels / 2) * nyq_percent).astype("int")
        k_space[0, nyq_samples] = fill_value
        k_space[0, -nyq_samples] = fill_value

    for nyq_percent in nyq_percents_y:
        nyq_samples = np.floor((num_channels / 2) * nyq_percent).astype("int")
        k_space[nyq_samples, 0] = fill_value
        k_space[-nyq_samples, 0] = fill_value

    channel_trend = np.abs(np.fft.ifft2(k_space))

    return channel_trend

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


def process_args_for_si(si_recording, lagc):
    """"""
    if type(lagc) == int:
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
        butter_kwargs = {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}

    if ntr_pad is None:
        ntr_pad = 0
    if lagc == "default":
        lagc = 300
    if lagc in [None, False]:
        lagc = 0

    return butter_kwargs, ntr_pad, lagc

def get_test_recording(num_channels=32, num_segments=2):
    """
    1500 and 2100 samples, [0.05, 0.07]
    """
    recording = generate_recording(num_channels=num_channels,
                                   durations=np.random.uniform(0.05, 0.07, num_segments))
    return recording

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


def get_ibl_si_data():
    """
    Set fixture to session to ensure origional data is not changed.
    """
    local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
    ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True)
    ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel

    si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
    si_recording = spre.scale(si_recording, dtype="float32")

    return [ibl_data, si_recording]


if __name__ == '__main__':
    test_add_trend_and_check_deleted(num_channels=384)  # TODO: this will fail with fixutres...