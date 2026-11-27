import csv

import pytest
import numpy as np

from spikeinterface.core import NumpyRecording, generate_recording
from spikeinterface.preprocessing import common_reference, highpass_filter
from spikeinterface.qualitycontrol import raw_data_quality_metrics, write_rms_summary_csvs
from spikeinterface.qualitycontrol.raw_data_quality_metrics import _generate_skipped_chunk_slices


class TestRawDataQualityMetrics:
    """
    Test the raw data quality metrics. The strategy is to build recordings
    from whole-cycle sines of known amplitude, so the expected RMS
    (amplitude / sqrt(2)) and Hann-windowed PSD can be computed analytically.
    """

    gain = 2.0
    offset = 10.0
    channel_amplitudes = np.array([1, 3])
    rms_quantile_fraction = np.array([1.2, 2.8])  # 10-90th percentiles of [1, 3]

    def make_recording(self, traces, sampling_frequency=2000):
        """
        Make a two-channel recording with non-unit gain and offset,
        so tests check that metrics are computed in microvolts.
        """
        recording = NumpyRecording(traces, sampling_frequency, channel_ids=["a", "b"])
        recording.set_channel_gains(self.gain)
        recording.set_channel_offsets(self.offset)
        return recording

    def make_sine(self, frequency, num_samples=2000, sampling_frequency=2000):
        """
        A sine of unit amplitude as a (num_samples, 1) column.
        """
        times = np.arange(num_samples) / sampling_frequency
        return np.sin(2 * np.pi * frequency * times)[:, None]

    def run_metrics(self, raw_ap, **kwargs):
        """
        Run the metrics serially with the raw recording also passed as
        the preprocessed recording, unless otherwise specified.
        """
        kwargs.setdefault("preprocessed_ap", raw_ap)
        return raw_data_quality_metrics(raw_ap, n_jobs=1, progress_bar=False, **kwargs)

    # #########################################################################
    # Tests
    # #########################################################################

    def test_metric_accuracy(self):
        """
        Test RMS, RMS quantiles, PSD and rolling RMS against analytical values
        for every recording type and segment. Each recording and segment is
        scaled differently to check results are not mixed up. A DC offset is
        added to check it is removed before RMS computation.
        """
        frequencies = [100, 200]
        sines = np.hstack([self.make_sine(f) for f in frequencies]) * self.channel_amplitudes

        scales = {"raw_ap": 1, "preprocessed_ap": 2, "raw_lfp": 3}
        segment_scales = [1, 2]
        recordings = {
            name: self.make_recording([sines * scale * seg_scale + 20 for seg_scale in segment_scales])
            for name, scale in scales.items()
        }

        results = self.run_metrics(
            **recordings,
            chunk_size_s=1.0,
            num_chunks=1,
            rolling_rms_window_size_s=1.0,
            compute_rolling_rms_for="all",
        )

        for name, scale in scales.items():
            summary = results["chunked_results"][name]
            rolling = results["rolling_rms"][name]

            assert np.array_equal(summary["channel_ids"], ["a", "b"])

            for segment_index, seg_scale in enumerate(segment_scales):
                amplitude_uv = self.channel_amplitudes * scale * seg_scale * self.gain
                expected_rms = amplitude_uv / np.sqrt(2)

                assert np.allclose(summary["rms_median"][segment_index], expected_rms, rtol=2e-6)
                assert np.allclose(
                    summary["rms_quantile"][segment_index],
                    self.rms_quantile_fraction * scale * seg_scale * self.gain / np.sqrt(2),
                    rtol=2e-6,
                )
                assert np.allclose(rolling["rms"][segment_index], expected_rms[None, :], rtol=2e-6)
                assert np.allclose(rolling["times"][segment_index], [1999 / 4000])

                # A bin-centered sine under a Hann window spreads its power
                # (A^2 / 2) over three 1 Hz bins in the ratio 1/6, 2/3, 1/6.
                assert np.array_equal(summary["freqs"][segment_index], np.arange(1001))
                expected_psd = np.zeros((1001, 2))
                for channel, frequency in enumerate(frequencies):
                    expected_psd[frequency - 1 : frequency + 2, channel] = (
                        amplitude_uv[channel] ** 2 / 2 * np.array([1 / 6, 2 / 3, 1 / 6])
                    )
                assert np.allclose(summary["psd"][segment_index], expected_psd, rtol=2e-6, atol=1e-9)

    def test_aggregation_over_chunks(self):
        """
        Test that the RMS is the median over chunks (not the mean) and
        the PSD is the mean over chunks. Three chunks with amplitude scales
        1, 7 and 3 give a median of 3 but mean of ~3.67.
        """
        chunk_scales = [1, 7, 3]
        sine = self.make_sine(100) * self.channel_amplitudes
        recording = self.make_recording(np.concatenate([sine * scale for scale in chunk_scales]))

        results = self.run_metrics(recording, chunk_size_s=1.0, num_chunks=3, compute_rolling_rms_for=())
        summary = results["chunked_results"]["raw_ap"]

        median_scale = np.median(chunk_scales)
        assert np.allclose(summary["rms_median"][0], self.channel_amplitudes * self.gain * median_scale / np.sqrt(2))
        assert np.allclose(
            summary["rms_quantile"][0], self.rms_quantile_fraction * self.gain * median_scale / np.sqrt(2)
        )

        # With 1 Hz bins, summing the density gives the total power, averaged over chunks.
        mean_power = (self.channel_amplitudes * self.gain) ** 2 / 2 * np.mean(np.square(chunk_scales))
        assert np.allclose(summary["psd"][0].sum(axis=0), mean_power)

    @pytest.mark.parametrize(
        "first_chunk_time_s, chunk_size_s, num_chunks, expected",
        [
            (0, 1, 2, [(0, 0, 1000), (0, 2000, 3000), (1, 0, 1000), (1, 1500, 2500)]),
            (
                0.5,
                0.5,
                3,
                [(0, 500, 1000), (0, 1500, 2000), (0, 2500, 3000), (1, 500, 1000), (1, 1250, 1750), (1, 2000, 2500)],
            ),
            (0.25, 0.25, 1, [(0, 250, 500), (1, 250, 500)]),
        ],
    )
    def test_generate_skipped_chunk_slices(self, first_chunk_time_s, chunk_size_s, num_chunks, expected):
        """
        Test chunks start at `first_chunk_time_s`, the last chunk ends on the
        final sample and the rest are spaced uniformly, for segments of different lengths.
        """
        recording = self.make_recording([np.zeros((3000, 2)), np.zeros((2500, 2))], sampling_frequency=1000)

        slices = _generate_skipped_chunk_slices(recording, chunk_size_s, num_chunks, first_chunk_time_s)

        assert slices == expected

    @pytest.mark.parametrize(
        "window_size_s, expected_times",
        [
            (1.0, [[0.4995, 1.4995, 2.4995], [0.4995, 1.4995, 2.2495]]),
            (2.0, [[0.9995, 2.4995], [0.9995, 2.2495]]),
        ],
    )
    def test_rolling_rms_windows(self, window_size_s, expected_times):
        """
        Test the rolling RMS uses consecutive windows of `rolling_rms_window_size_s`,
        that a shorter final window is kept and times are the window centers.
        """
        recording = self.make_recording([np.zeros((3000, 2)), np.zeros((2500, 2))], sampling_frequency=1000)

        results = self.run_metrics(recording, chunk_size_s=0.5, num_chunks=2, rolling_rms_window_size_s=window_size_s)
        rolling = results["rolling_rms"]["preprocessed_ap"]

        for segment_index, times in enumerate(expected_times):
            assert rolling["rms"][segment_index].shape == (len(times), 2)
            assert np.allclose(rolling["times"][segment_index], times)

    @pytest.mark.parametrize(
        "compute_rolling_rms_for, with_lfp, expected",
        [
            (None, True, {"preprocessed_ap"}),
            ((), True, set()),
            (("raw_lfp",), True, {"raw_lfp"}),
            ("all", False, {"raw_ap", "preprocessed_ap"}),
            ("all", True, {"raw_ap", "preprocessed_ap", "raw_lfp"}),
        ],
    )
    def test_rolling_rms_selection(self, compute_rolling_rms_for, with_lfp, expected):
        """
        Test the rolling RMS is only computed for the requested recordings (`None`
        uses the default), while the chunked metrics are always computed for all.
        """
        recording = self.make_recording(np.zeros((2000, 2)))
        kwargs = dict(chunk_size_s=1.0, num_chunks=1, rolling_rms_window_size_s=1.0)
        if with_lfp:
            kwargs["raw_lfp"] = recording
        if compute_rolling_rms_for is not None:
            kwargs["compute_rolling_rms_for"] = compute_rolling_rms_for

        results = self.run_metrics(recording, **kwargs)

        expected_chunked = {"raw_ap", "preprocessed_ap"} | ({"raw_lfp"} if with_lfp else set())
        assert set(results["rolling_rms"]) == expected
        assert set(results["chunked_results"]) == expected_chunked
        for summary in results["chunked_results"].values():
            assert set(summary) == {"rms_median", "rms_quantile", "psd", "freqs", "channel_ids"}

    def test_default_preprocessing(self):
        """
        Test that when `preprocessed_ap` is not given, the metrics match
        those of a 300 Hz highpass filter followed by common median reference.
        """
        recording = generate_recording(num_channels=4, durations=[2.0], seed=0)
        kwargs = dict(chunk_size_s=0.5, num_chunks=2, compute_rolling_rms_for=())

        results = self.run_metrics(recording, preprocessed_ap=None, **kwargs)

        expected_preprocessed = common_reference(highpass_filter(recording, freq_min=300), operator="median")
        expected = self.run_metrics(recording, preprocessed_ap=expected_preprocessed, **kwargs)

        summary = results["chunked_results"]["preprocessed_ap"]
        expected_summary = expected["chunked_results"]["preprocessed_ap"]
        for metric in ["rms_median", "rms_quantile", "psd"]:
            assert np.allclose(summary[metric][0], expected_summary[metric][0])
        assert not np.allclose(summary["rms_median"][0], results["chunked_results"]["raw_ap"]["rms_median"][0])

    def test_input_validation(self):
        """
        Test errors are raised for `raw_lfp` rolling RMS without an LFP recording,
        chunking arguments in `job_kwargs` and segments too short for the requested chunks.
        """
        recording = self.make_recording(np.zeros((2000, 2)))

        with pytest.raises(ValueError, match="requires raw_lfp"):
            self.run_metrics(recording, num_chunks=1, compute_rolling_rms_for=("raw_lfp",))

        with pytest.raises(ValueError, match="sets the chunk size internally"):
            self.run_metrics(recording, num_chunks=1, chunk_duration=1.0)

        with pytest.raises(ValueError, match="shorter than chunk_size_s"):
            _generate_skipped_chunk_slices(recording, chunk_size_s=2.0, num_chunks=1)

        with pytest.raises(ValueError, match="not long enough"):
            _generate_skipped_chunk_slices(recording, chunk_size_s=0.5, num_chunks=3, first_chunk_time_s=0.5)

    def test_write_rms_summary_csvs(self, tmp_path):
        """
        Test one CSV is written per recording, with a summary row of RMS quantiles
        followed by per-channel median RMS. Multi-segment columns get a segment suffix.
        """
        chunked_results = {
            "raw_ap": {
                "channel_ids": np.array(["a", "b"]),
                "rms_median": [np.array([1.0, 2.0])],
                "rms_quantile": [np.array([1.1, 1.9])],
            },
            "raw_lfp": {
                "channel_ids": np.array(["a", "b"]),
                "rms_median": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                "rms_quantile": [np.array([1.1, 1.9]), np.array([3.1, 3.9])],
            },
        }

        write_rms_summary_csvs(tmp_path / "csvs", chunked_results)

        with open(tmp_path / "csvs" / "raw_ap.csv", newline="") as file:
            assert list(csv.reader(file)) == [
                ["channel_id", "rms_median", "rms_quantile_10", "rms_quantile_90"],
                ["all_channels", "", "1.1", "1.9"],
                ["a", "1.0", "", ""],
                ["b", "2.0", "", ""],
            ]

        with open(tmp_path / "csvs" / "raw_lfp.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        assert rows[0]["rms_quantile_90_seg_1"] == "3.9"
        assert [row["rms_median_seg_1"] for row in rows[1:]] == ["3.0", "4.0"]
