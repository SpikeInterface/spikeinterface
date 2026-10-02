import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing import plot_raw_data_quality_metrics, raw_data_quality_metrics

REAL_DATA = True
REAL_DATA_DURATION_S = 20 * 60
DANDI_S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/a8f/800/a8f8003e-4483-4b50-8a45-91ac5971f5d5"
REAL_DATA_FOLDER = Path(__file__).parent / "cache_folder" / "dandi_000939_A3702_first_20_minutes"


def load_real_recording():
	from spikeinterface import load
	from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor

	class NwbRecordingWithoutChannelLocations(NwbRecordingExtractor):
		def set_channel_locations(self, locations):
			pass

	if REAL_DATA_FOLDER.exists():
		print(f"Loading cached DANDI recording from {REAL_DATA_FOLDER}")
		return load(REAL_DATA_FOLDER)

	print("Inspecting DANDI 000939 electrical series...")
	series_paths = NwbRecordingExtractor.fetch_available_electrical_series_paths(
		file_path=DANDI_S3_URL,
		stream_mode="remfile",
	)
	non_lfp_paths = [path for path in series_paths if "lfp" not in path.lower()]
	electrical_series_path = (non_lfp_paths or series_paths)[0]
	remote_recording = NwbRecordingWithoutChannelLocations(
		file_path=DANDI_S3_URL,
		stream_mode="remfile",
		electrical_series_path=electrical_series_path,
		load_channel_properties=False,
	)

	end_frame = min(
		round(REAL_DATA_DURATION_S * remote_recording.get_sampling_frequency()),
		remote_recording.get_num_samples(segment_index=0),
	)
	recording = remote_recording.frame_slice(start_frame=0, end_frame=end_frame)
	estimated_size_gb = end_frame * recording.get_num_channels() * recording.get_dtype().itemsize / 1e9
	print(
		f"Downloading {end_frame / recording.get_sampling_frequency() / 60:.1f} minutes "
		f"from {electrical_series_path} (approximately {estimated_size_gb:.1f} GB)..."
	)
	REAL_DATA_FOLDER.parent.mkdir(parents=True, exist_ok=True)
	return recording.save(format="binary", folder=REAL_DATA_FOLDER, overwrite=True)


def make_realistic_segment(segment_index, sampling_frequency, duration_s, num_channels, rng):
	num_samples = round(duration_s * sampling_frequency)
	times = np.arange(num_samples) / sampling_frequency
	channel_indices = np.arange(num_channels)

	noise_levels = np.linspace(7.0, 14.0, num_channels)
	if segment_index == 1:
		noise_levels[-2:] *= 2.5
	traces = rng.normal(size=(num_samples, num_channels)) * noise_levels

	phases = channel_indices * 0.35
	traces += 22 * np.sin(2 * np.pi * 8 * times[:, None] + phases)
	traces += 7 * np.sin(2 * np.pi * 40 * times[:, None] + phases / 2)
	traces += 5 * np.sin(2 * np.pi * 60 * times)[:, None]
	traces += 12 * np.sin(2 * np.pi * 0.08 * times)[:, None]

	waveform_times = np.arange(-5, 9) / sampling_frequency
	spike_waveform = -75 * np.exp(-(waveform_times / 0.00045) ** 2)
	spike_waveform += 24 * np.exp(-((waveform_times - 0.0011) / 0.0007) ** 2)
	for channel_index in range(num_channels):
		spike_count = rng.poisson((3.0 + channel_index * 0.4) * duration_s)
		spike_samples = rng.integers(10, num_samples - 10, size=spike_count)
		impulses = np.zeros(num_samples)
		impulses[spike_samples] = 1
		spike_signal = np.convolve(impulses, spike_waveform, mode="same")
		traces[:, channel_index] += spike_signal
		if channel_index + 1 < num_channels:
			traces[:, channel_index + 1] += 0.25 * spike_signal

	for artifact_time in (95.0, 215.0):
		artifact_start = round((artifact_time + 4 * segment_index) * sampling_frequency)
		artifact_size = round(0.35 * sampling_frequency)
		artifact = np.hanning(artifact_size)[:, None] * rng.normal(0, 180, (1, num_channels))
		traces[artifact_start : artifact_start + artifact_size] += artifact

	return traces.astype("float32")

if REAL_DATA:
	recording = load_real_recording()
else:
	sampling_frequency = 2_000.0
	duration_s = 301.0
	num_channels = 8
	rng = np.random.default_rng(seed=0)
	traces_list = [
		make_realistic_segment(segment_index, sampling_frequency, duration_s, num_channels, rng)
		for segment_index in range(2)
	]
	recording = NumpyRecording(traces_list, sampling_frequency=sampling_frequency)

results = raw_data_quality_metrics(recording, n_jobs=10)
for segment_index in range(recording.get_num_segments()):
	plot_raw_data_quality_metrics(results["quick"], results["rolling_rms"], segment_index=segment_index)
plt.show()
