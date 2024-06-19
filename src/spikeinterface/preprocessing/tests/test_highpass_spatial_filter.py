import pytest
import os
import numpy as np
from copy import deepcopy

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface.core import generate_recording
import spikeinterface.widgets as sw
import importlib.util

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("neurodsp") is not None or importlib.util.find_spec("spikeglx") or ON_GITHUB,
    reason="Only local. Requires ibl-neuropixel install",
)
@pytest.mark.parametrize("lagc", [False, 1, 300])
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
    import spikeglx
    import neurodsp.voltage as voltage

    options = dict(lagc=lagc, ntr_pad=25, ntr_tap=50, butter_kwargs=None)
    print(options)

    ibl_data, si_recording = get_ibl_si_data()

    si_filtered, _ = run_si_highpass_filter(si_recording, **options)

    ibl_filtered = run_ibl_highpass_filter(ibl_data.copy(), **options)

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

    assert np.allclose(
        si_filtered, ibl_filtered * 1e6, atol=1e-01, rtol=0
    )  # the differences are entired due to scaling on data load.


@pytest.mark.parametrize("ntr_pad", [None, 0, 31])
@pytest.mark.parametrize("ntr_tap", [None, 5])
@pytest.mark.parametrize("lagc", [False, 300, 1232])
@pytest.mark.parametrize("butter_kwargs", [None, {"N": 5, "Wn": 0.12}])
@pytest.mark.parametrize("num_channels", [32, 64])
def test_highpass_spatial_filter_synthetic_data(num_channels, ntr_pad, ntr_tap, lagc, butter_kwargs):
    """
    Generate a short recording, run it through SI and IBL version, check outputs match. Used to
    check many combinations of possible inputs.
    """
    options = dict(lagc=lagc, ntr_pad=ntr_pad, ntr_tap=ntr_tap, butter_kwargs=butter_kwargs)

    durations = [2, 2]
    rng = np.random.RandomState(seed=100)
    si_recording = generate_recording(num_channels=num_channels, durations=durations)

    _, si_highpass_spatial_filter = run_si_highpass_filter(si_recording, get_traces=False, **options)
    frames = [(0, 500), (30000, 33000), (57000, 60000)]
    # only test trace retrieval here
    for seg in range(si_recording.get_num_segments()):
        for frame in frames:
            raw_traces = si_recording.get_traces(segment_index=seg, start_frame=frame[0], end_frame=frame[1])
            si_filtered = si_highpass_spatial_filter.get_traces(
                segment_index=seg, start_frame=frame[0], end_frame=frame[1]
            )
            assert raw_traces.shape == si_filtered.shape


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_dtype_stability(dtype):
    """
    Check that the dtype of the recording and
    output data is as expected, as data is cast to float32
    during filtering.
    """
    num_chan = 32
    si_recording = generate_recording(num_channels=num_chan, durations=[2])
    si_recording.set_property("gain_to_uV", np.ones(num_chan))
    si_recording.set_property("offset_to_uV", np.ones(num_chan))
    si_recording = spre.astype(si_recording, dtype)

    assert si_recording.dtype == dtype

    highpass_spatial_filter = spre.highpass_spatial_filter(si_recording, n_channel_pad=2)

    assert highpass_spatial_filter.dtype == dtype

    filtered_data_unscaled = highpass_spatial_filter.get_traces(return_scaled=False)

    assert filtered_data_unscaled.dtype == dtype

    filtered_data_scaled = highpass_spatial_filter.get_traces(return_scaled=True)

    assert filtered_data_scaled.dtype == np.float32


# ----------------------------------------------------------------------------------------------------------------------
# Test Utils
# ----------------------------------------------------------------------------------------------------------------------


def get_ibl_si_data():
    """
    Set fixture to session to ensure origional data is not changed.
    """
    import spikeglx

    local_path = si.download_dataset(remote_path="spikeglx/Noise4Sam_g0")
    ibl_recording = spikeglx.Reader(
        local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True
    )
    ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel

    si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
    si_recording = spre.astype(si_recording, dtype="float32")

    return ibl_data, si_recording


def process_args_for_si(si_recording, lagc):
    """"""
    if isinstance(lagc, bool) and not lagc:
        agc_window_length_s = None
        apply_agc = False
    else:
        assert lagc > 0
        ts = 1 / si_recording.sampling_frequency
        window_s = lagc * ts
        agc_window_length_s = window_s
        apply_agc = True
    si_agc_params = {"apply_agc": apply_agc, "agc_window_length_s": agc_window_length_s}

    return si_agc_params


def process_args_for_ibl(butter_kwargs, ntr_pad, lagc):
    """"""
    if butter_kwargs is None:
        butter_kwargs_ = {"N": 3, "Wn": 0.01}
    else:
        butter_kwargs_ = deepcopy(butter_kwargs)
    if "btype" not in butter_kwargs_:
        butter_kwargs_["btype"] = "highpass"

    if ntr_pad is None:
        ntr_pad = 0

    if lagc in [None, False]:
        lagc = 0

    return butter_kwargs_, ntr_pad, lagc


def run_si_highpass_filter(si_recording, ntr_pad, ntr_tap, lagc, butter_kwargs, get_traces=True):
    """"""
    si_lagc = process_args_for_si(si_recording, lagc)
    if butter_kwargs is not None:
        highpass_butter_order = butter_kwargs["N"]
        highpass_butter_wn = butter_kwargs["Wn"]
        butter_kwargs = dict(highpass_butter_order=highpass_butter_order, highpass_butter_wn=highpass_butter_wn)
    else:
        butter_kwargs = {}

    si_highpass_spatial_filter = spre.highpass_spatial_filter(
        si_recording, n_channel_pad=ntr_pad, n_channel_taper=ntr_tap, **si_lagc, **butter_kwargs
    )

    if get_traces:
        si_filtered = si_highpass_spatial_filter.get_traces(return_scaled=True)
    else:
        si_filtered = False

    return si_filtered, si_highpass_spatial_filter


def run_ibl_highpass_filter(ibl_data, ntr_pad, ntr_tap, lagc, butter_kwargs):
    butter_kwargs, ntr_pad, lagc = process_args_for_ibl(butter_kwargs, ntr_pad, lagc)

    ibl_filtered = voltage.kfilt(ibl_data, None, ntr_pad, ntr_tap, lagc, butter_kwargs).T

    return ibl_filtered


if __name__ == "__main__":
    test_highpass_spatial_filter_real_data(lagc=False)
    test_highpass_spatial_filter_synthetic_data(64, None, None, 1232, None)
