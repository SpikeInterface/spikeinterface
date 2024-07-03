import pytest
import numpy as np
import os

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface.core.generate import generate_recording
import importlib.util


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))
DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()


# -------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("neurodsp") is not None or importlib.util.find_spec("spikeglx") or ON_GITHUB,
    reason="Only local. Requires ibl-neuropixel install",
)
def test_compare_real_data_with_ibl():
    """
    Test SI implementation of bad channel interpolation against native IBL.

    Requires preprocessing scaled values to get close alignment (otherwise
    different on average by ~1.1)

    They are not exactly the same due to minor scaling differences (applying
    voltage.interpolate_bad_channel() with ibl_channel geometry  to
    si_scaled_recordin.get_traces(0) is also close to 1e-2.
    """
    # Download and load data
    import spikeglx
    import neurodsp.voltage as voltage

    local_path = si.download_dataset(remote_path="spikeglx/Noise4Sam_g0")
    si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
    ibl_recording = spikeglx.Reader(
        local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin", ignore_warnings=True
    )

    num_channels = si_recording.get_num_channels()
    bad_channel_indexes = np.random.choice(num_channels, 10, replace=False)
    bad_channel_ids = si_recording.channel_ids[bad_channel_indexes]
    si_recording = spre.scale(si_recording, dtype="float32")

    # interpolate SI
    si_interpolated_recording = spre.interpolate_bad_channels(si_recording, bad_channel_ids)

    # interpolate IBL
    ibl_bad_channel_labels = get_ibl_bad_channel_labels(num_channels, bad_channel_indexes)

    ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel
    si_interpolated = si_interpolated_recording.get_traces(return_scaled=True)
    ibl_interpolated = voltage.interpolate_bad_channels(
        ibl_data, ibl_bad_channel_labels, x=ibl_recording.geometry["x"], y=ibl_recording.geometry["y"]
    ).T

    if DEBUG:
        for bad_idx in bad_channel_indexes:
            fig, ax = plt.subplots()
            ax.plot(si_interpolated[:, bad_idx], label="SI")
            ax.plot(ibl_interpolated[:, bad_idx] * 1e6, label="IBL")
            ax.set_title(f"bad channel {bad_idx}")
            ax.legend()

    # compare
    assert np.allclose(ibl_interpolated * 1e6, si_interpolated, 1e-1)
    is_close = np.isclose(ibl_interpolated * 1e6, si_interpolated, 1e-5)
    assert np.mean(is_close) > 0.999


@pytest.mark.skipif(
    importlib.util.find_spec("neurodsp") is not None or importlib.util.find_spec("spikeglx") is not None,
    reason="Requires ibl-neuropixel install",
)
@pytest.mark.parametrize("num_channels", [32, 64])
@pytest.mark.parametrize("sigma_um", [1.25, 40])
@pytest.mark.parametrize("p", [0, -0.5, 1, 5])
@pytest.mark.parametrize("shanks", [4, 1])
def test_compare_input_argument_ranges_against_ibl(shanks, p, sigma_um, num_channels):
    """
    Perform an extended test across a range of function inputs to check
    IBL and SI interpolation results match.
    """
    import neurodsp.voltage as voltage

    recording = generate_recording(num_channels=num_channels, durations=[1])

    # distribute default probe locations across 4 shanks if set
    x = np.random.choice(shanks, num_channels)
    for idx, __ in enumerate(recording._properties["contact_vector"]):
        recording._properties["contact_vector"][idx][1] = x[idx]

    # generate random bad channel locations
    bad_channel_indexes = np.random.choice(num_channels, np.random.randint(1, int(num_channels / 5)), replace=False)
    bad_channel_ids = recording.channel_ids[bad_channel_indexes]

    # Run SI and IBL interpolation and check against eachother
    recording = spre.scale(recording, dtype="float32")
    si_interpolated_recording = spre.interpolate_bad_channels(recording, bad_channel_ids, sigma_um=sigma_um, p=p)
    si_interpolated = si_interpolated_recording.get_traces()

    ibl_bad_channel_labels = get_ibl_bad_channel_labels(num_channels, bad_channel_indexes)
    x, y = np.hsplit(recording.get_probe().contact_positions, 2)
    ibl_interpolated = voltage.interpolate_bad_channels(
        recording.get_traces().T, ibl_bad_channel_labels, x=x.ravel(), y=y.ravel(), p=p, kriging_distance_um=sigma_um
    ).T

    assert np.allclose(si_interpolated, ibl_interpolated, rtol=0, atol=1e-06)


def test_output_values():
    """
    Quick sanity check that the outputs are as expected. Settings all
    channels equally apart, the interpolated channel should be a linear
    combination of the non-bad channels, using arbitary sigma_um and p.

    Then, set the final channel to twice as far away as the rest of the
    other channels. Calculate the expected weights and check they
    match interpolation output.

    Checking the bad channel ts is a combination of
    the non-interpolated channels is also an implicit test
    these were not accidently changed.
    """
    recording = generate_recording(num_channels=5, durations=[1])
    bad_channel_indexes = np.array([0])
    bad_channel_ids = recording.channel_ids[bad_channel_indexes]

    new_probe_locs = [
        [5, 7, 3, 5, 5],  # 5 channels, a in the center ('bad channel', zero index)
        [5, 5, 5, 7, 3],
    ]  # all others equal distance away.
    # Overwrite the probe information with the new locations
    for idx, (x, y) in enumerate(zip(*new_probe_locs)):
        recording._properties["contact_vector"][idx][1] = x
        recording._properties["contact_vector"][idx][2] = y

    # Run interpolation in SI and check the interpolated channel
    # 0 is a linear combination of other channels
    recording = spre.scale(recording, dtype="float32")
    si_interpolated_recording = spre.interpolate_bad_channels(recording, bad_channel_ids, sigma_um=5, p=2)
    si_interpolated = si_interpolated_recording.get_traces()
    expected_ts = np.sum(si_interpolated[:, 1:] / 4, axis=1)

    assert np.allclose(si_interpolated[:, 0], expected_ts, rtol=0, atol=1e-06)

    # Shift the last channel position so that it is 4 units, rather than 2
    # away. Setting sigma_um = p = 1 allows easy calculation of the expected
    # weights.
    recording._properties["contact_vector"][-1][1] = 5
    recording._properties["contact_vector"][-1][2] = 9
    expected_weights = np.r_[np.tile(np.exp(-2), 3), np.exp(-4)]
    expected_weights /= np.sum(expected_weights)

    si_interpolated_recording = spre.interpolate_bad_channels(recording, bad_channel_indexes, sigma_um=1, p=1)
    si_interpolated = si_interpolated_recording.get_traces()

    expected_ts = si_interpolated[:, 1:] @ expected_weights

    assert np.allclose(si_interpolated[:, 0], expected_ts, rtol=0, atol=1e-06)


# -------------------------------------------------------------------------------
# Test Utils
# -------------------------------------------------------------------------------


def get_ibl_bad_channel_labels(num_channels, bad_channel_indexes):
    ibl_bad_channel_labels = np.zeros(num_channels)
    ibl_bad_channel_labels[bad_channel_indexes] = 1
    return ibl_bad_channel_labels


if __name__ == "__main__":
    test_compare_real_data_with_ibl()
    test_compare_input_argument_ranges_against_ibl(shanks=4, p=1, sigma_um=1.25, num_channels=32)
    test_output_values()
