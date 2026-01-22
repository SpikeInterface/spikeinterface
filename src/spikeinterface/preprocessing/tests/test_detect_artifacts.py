from spikeinterface.core import generate_recording, NumpyRecording
from spikeinterface.preprocessing import detect_artifact_periods, detect_saturation_periods
import numpy as np


def test_detect_artifact_periods():
    # one segment only
    rec = generate_recording(durations=[10.0, 10])
    artifacts = detect_artifact_periods(rec, method="envelope", 
                                        method_kwargs=dict(detect_threshold=5, freq_max=5.0),
                                        )



def test_detect_saturation_periods():

    import scipy.signal
    
    """
    TODO: NOTE: we have one sample before the saturation starts as we take the forward derivative for the velocity
                we have an extra sample after due to taking the diff on the final saturation mask
                this means we always take one sample before and one sample after the diff period, which is fine.
    """
    # num_chans = 384
    num_chans = 32
    sample_frequency = 30000
    chunk_size = 30000  # This value is critical to ensure hard-coded start / stops below
    job_kwargs = {"chunk_size": chunk_size}

    # cross a chunk boundary. Do not change without changing the below.
    sat_value = 1200
    rng = np.random.default_rng()
    data = rng.uniform(low=-0.5, high=0.5, size=(150000, num_chans)) * 10 * 1e-6

    # Design the Butterworth filter
    sos = scipy.signal.butter(N=3, Wn=12000 / (sample_frequency / 2), btype="low", output="sos")

    # Apply the filter to the data
    data_seg_1 = scipy.signal.sosfiltfilt(sos, data, axis=0)
    data_seg_2 = data_seg_1.copy()

    # Add test saturation at the start, end of recording
    # as well as across and within chunks (30k samples).
    # Two cases which are not tested are a single event
    # exactly on the border, as it makes testing complex
    # This was checked manually and any future breaking change
    # on this function would be extremely unlikely only to break this case.
    all_starts = np.array([0,    29950,  45123, 90005,  149500])
    all_stops =  np.array([1001, 30011,  45126, 90006,  149999])

    second_seg_offset = 1
    for start, stop in zip(all_starts, all_stops):
        data_seg_1[start : stop, :] = sat_value
        # differentiate the second segment for testing purposes
        data_seg_2[start : stop + second_seg_offset, :] = sat_value

    # this center the int16 around 0 and saturate on positive
    max_ = np.max(np.r_[data_seg_1.flatten(), data_seg_2.flatten()])
    gain =  max_ / 2**15
    offset = 0

    seg_1_int16 = np.clip(
        np.rint((data_seg_1 - offset) / gain),
        -32768, 32767
    ).astype(np.int16)
    seg_2_int16 = np.clip(
        np.rint((data_seg_2 - offset) / gain),
        -32768, 32767
    ).astype(np.int16)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(seg_1_int16[:, 0])
    # plt.show()

    recording = NumpyRecording([seg_1_int16, seg_2_int16], sample_frequency)
    recording.set_channel_gains(gain)
    recording.set_channel_offsets([offset] * num_chans)

    periods = detect_saturation_periods(
        recording, saturation_threshold_uV=sat_value * 0.98, voltage_per_sec_threshold=1e-8, job_kwargs=job_kwargs
    )

    seg_1_periods = periods[np.where(periods["segment_index"] == 0)]
    seg_2_periods = periods[np.where(periods["segment_index"] == 1)]

    # For the start times, all are one sample before the actual saturated
    # period starts because the derivative threshold is exceeded at one
    # sample before the saturation starts. Therefore this one-sample-offset
    # on the start times is an implicit test that the derivative
    # threshold is working properly.
    for seg_periods in [seg_1_periods, seg_2_periods]:
        assert seg_periods["start_sample_index"][0] == all_starts[0]
        assert np.array_equal(seg_periods["start_sample_index"][1:], np.array(all_starts)[1:] - 1)

    assert np.array_equal(seg_1_periods["end_sample_index"], np.array(all_stops))
    assert np.array_equal(seg_2_periods["end_sample_index"], np.array(all_stops) + second_seg_offset)

    # Just do a quick test that a threshold slightly over the sat value is not detected.
    # In this case we only see the derivative threshold detection. We do not play around with this
    # threshold because the derivative threshold is not easy to predict (the baseline sample is random).
    periods = detect_saturation_periods(
        recording,
        saturation_threshold_uV=sat_value * (1 / 0.98),
        voltage_per_sec_threshold=1e-8,
        job_kwargs=job_kwargs,
    )
    assert periods["start_sample_index"][0] == 1000
    assert periods["end_sample_index"][0] == 1001

    periods = detect_artifact_periods(
        recording,
        method="saturation",
        method_kwargs=dict(
                saturation_threshold_uV=sat_value * (1 / 0.98),
                voltage_per_sec_threshold=1e-8,
            ),
            job_kwargs=job_kwargs,
        )



if __name__ == "__main__":
    test_detect_artifact_periods()
    test_detect_saturation_periods()
