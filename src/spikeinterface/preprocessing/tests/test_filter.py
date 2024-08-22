import pytest

import numpy as np
from spikeinterface.core import generate_recording
from spikeinterface import NumpyRecording, set_global_tmp_folder

from spikeinterface.preprocessing import filter, bandpass_filter, notch_filter, causal_filter


class TestCausalFilter:
    """
    The only thing that is not tested (JZ, as of 23/07/2024) is the
    propagation of margin kwargs, these are general filter params
    and can be tested in an upcoming PR.
    """

    @pytest.fixture(scope="session")
    def recording_and_data(self):
        recording = generate_recording(durations=[1])
        raw_data = recording.get_traces()

        return (recording, raw_data)

    def test_causal_filter_main_kwargs(self, recording_and_data):
        """
        Perform a test that expected output is returned under change
        of all key filter-related kwargs. First run the filter in
        the forward direction with options and compare it
        to the expected output from scipy.

        Next, change every filter-related kwarg and set in the backwards
        direction. Again check it matches expected scipy output.
        """
        from scipy.signal import lfilter, sosfilt

        recording, raw_data = recording_and_data

        # First, check in the forward direction with
        # the default set of kwargs
        options = self._get_filter_options()

        sos = self._run_iirfilter(options, recording)

        test_data = sosfilt(sos, raw_data, axis=0)
        test_data.astype(recording.dtype)

        filt_data = causal_filter(recording, direction="forward", **options, margin_ms=0).get_traces()

        assert np.allclose(test_data, filt_data, rtol=0, atol=1e-6)

        # Then, change all kwargs to ensure they are propagated
        # and check the backwards version.
        options["band"] = [671]
        options["btype"] = "highpass"
        options["filter_order"] = 8
        options["ftype"] = "bessel"
        options["filter_mode"] = "ba"
        options["dtype"] = np.float16

        b, a = self._run_iirfilter(options, recording)

        flip_raw = np.flip(raw_data, axis=0)
        test_data = lfilter(b, a, flip_raw, axis=0)
        test_data = np.flip(test_data, axis=0)
        test_data = test_data.astype(options["dtype"])

        filt_data = causal_filter(recording, direction="backward", **options, margin_ms=0).get_traces()

        assert np.allclose(test_data, filt_data, rtol=0, atol=1e-6)

    def test_causal_filter_custom_coeff(self, recording_and_data):
        """
        A different path is taken when custom coeff is selected.
        Therefore, explicitly test the expected outputs are obtained
        when passing custom coeff, under the "ba" and "sos" conditions.
        """
        from scipy.signal import lfilter, sosfilt

        recording, raw_data = recording_and_data

        options = self._get_filter_options()
        options["filter_mode"] = "ba"
        options["coeff"] = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]))

        # Check the custom coeff are propagated in both modes.
        # First, in "ba" mode
        test_data = lfilter(options["coeff"][0], options["coeff"][1], raw_data, axis=0)
        test_data = test_data.astype(recording.get_dtype())

        filt_data = causal_filter(recording, direction="forward", **options, margin_ms=0).get_traces()

        assert np.allclose(test_data, filt_data, rtol=0, atol=1e-6, equal_nan=True)

        # Next, in "sos" mode
        options["filter_mode"] = "sos"
        options["coeff"] = np.ones((2, 6))

        test_data = sosfilt(options["coeff"], raw_data, axis=0)
        test_data = test_data.astype(recording.get_dtype())

        filt_data = causal_filter(recording, direction="forward", **options, margin_ms=0).get_traces()

        assert np.allclose(test_data, filt_data, rtol=0, atol=1e-6, equal_nan=True)

    def test_causal_kwarg_error_raised(self, recording_and_data):
        """
        Test that passing the "forward-backward" direction results in
        an error. It is is critical this error is raised,
        otherwise the filter will no longer be causal.
        """
        recording, raw_data = recording_and_data

        with pytest.raises(BaseException) as e:
            filt_data = causal_filter(recording, direction="forward-backward")

    def _run_iirfilter(self, options, recording):
        """
        Convenience function to convert Si kwarg
        names to Scipy.
        """
        from scipy.signal import iirfilter

        return iirfilter(
            N=options["filter_order"],
            Wn=options["band"],
            btype=options["btype"],
            ftype=options["ftype"],
            output=options["filter_mode"],
            fs=recording.get_sampling_frequency(),
        )

    def _get_filter_options(self):
        return {
            "band": [300.0, 6000.0],
            "btype": "bandpass",
            "filter_order": 5,
            "ftype": "butter",
            "filter_mode": "sos",
            "coeff": None,
        }


def test_filter():
    rec = generate_recording()
    rec = rec.save()

    rec2 = bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)

    # compute by chunk
    rec2_cached0 = rec2.save(chunk_size=100000, verbose=False, progress_bar=True)

    # compute by chunkf with joblib
    rec2_cached1 = rec2.save(total_memory="10k", n_jobs=4, verbose=True)

    # compute once
    rec2_cached2 = rec2.save(verbose=False)

    trace0 = rec2.get_traces(segment_index=0)
    trace1 = rec2_cached1.get_traces(segment_index=0)

    # other filtering types
    rec3 = filter(rec, band=500.0, btype="highpass", filter_mode="ba", filter_order=2)
    rec4 = notch_filter(rec, freq=3000, q=30, margin_ms=5.0)
    rec5 = causal_filter(rec, direction="forward")
    rec6 = causal_filter(rec, direction="backward")

    # filter from coefficients
    from scipy.signal import iirfilter

    coeff = iirfilter(8, [0.02, 0.4], rs=30, btype="band", analog=False, ftype="cheby2", output="sos")
    rec5 = filter(rec, coeff=coeff, filter_mode="sos")

    # compute by chunk
    rec5_cached0 = rec5.save(chunk_size=100000, verbose=False, progress_bar=True)

    trace50 = rec5.get_traces(segment_index=0)
    trace51 = rec5_cached0.get_traces(segment_index=0)

    assert np.allclose(rec.get_times(0), rec2.get_times(0))

    # reflect padding test
    rec6 = bandpass_filter(rec, freq_min=300.0, freq_max=6000.0, add_reflect_padding=True)
    rec6_cached = rec6.save(chunk_size=150000, verbose=False, progress_bar=True)
    trace0 = rec6.get_traces(segment_index=0)
    trace1 = rec6_cached.get_traces(segment_index=0)

    print(trace0.shape, trace1.shape)
    print(np.abs(trace0 - trace1).max())

    assert np.allclose(trace0, trace1)


def test_filter_unsigned():
    traces = np.random.randint(1, 1000, (5000, 4), dtype="uint16")
    rec = NumpyRecording(traces_list=traces, sampling_frequency=1000)
    rec = rec.save()

    rec2 = bandpass_filter(rec, freq_min=10.0, freq_max=300.0)
    assert not np.issubdtype(rec2.get_dtype(), np.unsignedinteger)
    traces2 = rec2.get_traces()
    assert not np.issubdtype(traces2.dtype, np.unsignedinteger)

    # notch filter note supported for unsigned
    with pytest.raises(TypeError):
        rec3 = notch_filter(rec, freq=300.0, q=10)

    # this is ok
    rec3 = notch_filter(rec, freq=300.0, q=10, dtype="float32")


@pytest.mark.skip("OpenCL not tested")
def test_filter_opencl():
    rec = generate_recording(
        num_channels=256,
        # num_channels = 32,
        sampling_frequency=30000.0,
        durations=[
            100.325,
        ],
        # durations = [10.325, 3.5],
    )
    rec = rec.save(total_memory="100M", n_jobs=1, progress_bar=True)

    print(rec.get_dtype())

    rec_filtered = filter(rec, engine="scipy")
    rec_filtered = rec_filtered.save(chunk_size=1000, progress_bar=True, n_jobs=30)

    rec2 = filter(rec, engine="opencl")
    rec2_cached0 = rec2.save(chunk_size=1000, verbose=False, progress_bar=True, n_jobs=1)
    # rec2_cached0 = rec2.save(chunk_size=1000,verbose=False, progress_bar=True, n_jobs=4)

    # import matplotlib.pyplot as plt
    # from spikeinterface.widgets import plot_traces
    # plot_traces(rec, segment_index=0)
    # plot_traces(rec_filtered, segment_index=0)
    # plot_traces(rec2_cached0, segment_index=0)
    # plt.show()


if __name__ == "__main__":
    test_filter()
    test_filter_unsigned()
