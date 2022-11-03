import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import pytest
import numpy as np

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
def test_interpolate_bad_channels():
    """
    Test SI implementation of bad channel interpolation against native IBL.

    Requires preprocessing scaled values to get close alignment (otherwise
    different on average by ~1.1)

    They are not exactly the same due to minor scaling differences (applying
    voltage.interpolate_bad_channel() with ibl_channel geometry  to
    si_scaled_recordin.get_traces(0) is also close to 1e-2.
    """
    # Download and load data
    local_path = si.download_dataset(remote_path='spikeglx/Noise4Sam_g0')
    si_recording = se.read_spikeglx(local_path, stream_id="imec0.ap")
    ibl_recording = spikeglx.Reader(local_path / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin",
                                    ignore_warnings=True)

    bad_channel_indexes = np.random.choice(si_recording.get_num_channels(),
                                           10, replace=False)
    si_recording = spre.scale(si_recording, dtype="float32")

    # interpolate SI
    si_interpolated_recording = spre.interpolate_bad_channels(si_recording,
                                                              bad_channel_indexes)

    # interpolate IBL
    ibl_bad_channel_labels = np.zeros(si_recording.get_num_channels())
    ibl_bad_channel_labels[bad_channel_indexes] = 1

    ibl_data = ibl_recording.read(slice(None), slice(None), sync=False)[:, :-1].T  # cut sync channel
    si_interpolated = si_interpolated_recording.get_traces(return_scaled=True)
    ibl_interpolated = voltage.interpolate_bad_channels(ibl_data,
                                                        ibl_bad_channel_labels,
                                                        x=ibl_recording.geometry["x"],
                                                        y=ibl_recording.geometry["y"]).T

    if DEBUG:
        for bad_idx in bad_channel_indexes:
            fig, ax = plt.subplots()
            ax.plot(si_interpolated[:, bad_idx], label="SI")
            ax.plot(ibl_interpolated[:, bad_idx]*1e6, label="IBL")
            ax.set_title(f"bad channel {bad_idx}")
            ax.legend()

    # compare
    assert np.allclose(ibl_interpolated*1e6, si_interpolated, 1e-1)
    is_close = np.isclose(ibl_interpolated*1e6, si_interpolated, 1e-5)
    assert np.mean(is_close) > 0.999

if __name__ == '__main__':
    test_interpolate_bad_channels()
