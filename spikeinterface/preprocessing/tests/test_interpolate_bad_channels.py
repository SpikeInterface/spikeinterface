import spikeinterface as si
import spikeinterface.preprocessing as sipp
import spikeinterface.extractors as se
import copy
import numpy as np

try:
    import spikeglx
    import neurodsp.voltage as voltage
except ImportError:
    raise ImportError("Requires ibl-neuropixel dev install (pip install -e .) from inside cloned repo."
                      "https://github.com/int-brain-lab/ibl-neuropixel")

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
                                           100, replace=False)

    # interpolate SI
    si_scaled_recording = sipp.ScaleRecording(si_recording,
                                              gain=si_recording.get_property("gain_to_uV"),
                                              offset=si_recording.get_property("offset_to_uV"))

    si_interpolated_recording = sipp.interpolate_bad_channels(si_scaled_recording,
                                                              bad_channel_indexes)
    si_interpolated = si_interpolated_recording.get_traces(0)

    # interpolate IBL
    ibl_bad_channel_labels = np.zeros(si_recording.get_num_channels())
    ibl_bad_channel_labels[bad_channel_indexes] = 1

    ibl_data = ibl_recording.read()[0].T[:-1, :]  # cut sync channel

    ibl_interpolated = voltage.interpolate_bad_channels(ibl_data,
                                                        ibl_bad_channel_labels,
                                                        x=ibl_recording.geometry["x"],
                                                        y=ibl_recording.geometry["y"])

    # compare
    assert np.allclose(ibl_interpolated.T*1e6, si_interpolated, 1e-2)
    is_close = np.isclose(ibl_interpolated.T*1e6, si_interpolated, 1e-5)
    assert np.mean(is_close) > 0.999

if __name__ == '__main__':
    test_interpolate_bad_channels()
