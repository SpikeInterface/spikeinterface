"""
This is a temporary LLM-generated script for testing the raw data quality metrics on some real data.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing import plot_raw_data_quality_metrics, raw_data_quality_metrics

REAL_DATA = True
REAL_DATA_DURATION_S = 10 * 60
# Neuropixels recording in macaque V1 during presentation of static gratings.
# DANDI 001290, version 0.260827.1449, subject L11104 (CC-BY-4.0).
# https://dandiarchive.org/dandiset/001290/0.260827.1449
DANDI_S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/e31/54b/e3154bc7-9510-436e-b6bb-fa181d70dc08"
DANDI_ASSET_PATH = "sub-L11104/sub-L11104_ses-None_ecephys.nwb"
DANDI_FILE_SIZE = 7_378_434_120
DANDI_SHA256 = "720a532ec0ef60ee82c9b4de9c6013a3b6f9f7642893d881a712651fd5df91f8"
REAL_DATA_FOLDER = Path(__file__).parent / "cache_folder" / "dandi_001290_L11104"
DANDI_NWB_PATH = REAL_DATA_FOLDER / Path(DANDI_ASSET_PATH).name


def download_neuropixels_recording():
    """Download the NWB file once, resuming partial downloads and verifying its checksum."""
    import hashlib
    import requests
    from tqdm.auto import tqdm

    if DANDI_NWB_PATH.exists():
        if DANDI_NWB_PATH.stat().st_size != DANDI_FILE_SIZE:
            raise ValueError(f"Unexpected size for cached recording: {DANDI_NWB_PATH}")
        return DANDI_NWB_PATH

    REAL_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    partial_path = DANDI_NWB_PATH.with_suffix(".nwb.part")
    downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    if downloaded > DANDI_FILE_SIZE:
        raise ValueError(f"Partial download is larger than the expected file: {partial_path}")

    if downloaded < DANDI_FILE_SIZE:
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        with requests.get(DANDI_S3_URL, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if downloaded and (
                response.status_code != 206
                or not response.headers.get("Content-Range", "").startswith(f"bytes {downloaded}-")
            ):
                raise RuntimeError("Server did not honor the download resume offset")
            with partial_path.open("ab" if downloaded else "wb") as output, tqdm(
                total=DANDI_FILE_SIZE, initial=downloaded, unit="B", unit_scale=True,
                desc="Downloading DANDI Neuropixels NWB", mininterval=10,
            ) as progress:
                for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                    output.write(block)
                    progress.update(len(block))

    if partial_path.stat().st_size != DANDI_FILE_SIZE:
        raise RuntimeError("Incomplete download; rerun to resume")
    print("Verifying DANDI SHA-256 checksum...", flush=True)
    digest = hashlib.sha256()
    with partial_path.open("rb") as downloaded_file:
        for block in iter(lambda: downloaded_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != DANDI_SHA256:
        raise ValueError(f"DANDI checksum mismatch: {partial_path}")
    partial_path.rename(DANDI_NWB_PATH)
    return DANDI_NWB_PATH


def load_real_recordings():
    """Load matching AP and LFP streams from the downloaded Neuropixels recording."""
    from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor

    file_path = download_neuropixels_recording()
    recordings = []
    for series_name in ("ElectricalSeriesAPImec", "ElectricalSeriesLFImec"):
        recording = NwbRecordingExtractor(
            file_path=file_path,
            electrical_series_path=f"acquisition/{series_name}",
        )
        end_frame = min(
            round(REAL_DATA_DURATION_S * recording.get_sampling_frequency()),
            recording.get_num_samples(segment_index=0),
        )
        recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
        print(f"{series_name}: {recording}")
        recordings.append(recording)
    return tuple(recordings)


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

if __name__ == "__main__":
	import spikeinterface.full as si

	if REAL_DATA:
		recording, lfp_recording = load_real_recordings()
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
		lfp_recording = si.bandpass_filter(recording, freq_min=1, freq_max=300)

	bad_channels, bad_channel_ids = si.detect_bad_channels(recording)
	recording = recording.remove_channels(bad_channels)

#	bad_channels, bad_channel_ids = si.detect_bad_channels(lfp_recording)
#	lfp_recording = lfp_recording.remove_channels(bad_channels)

	results = raw_data_quality_metrics(
		recording,
		raw_lfp_recording=lfp_recording,
		n_jobs=10,
		folder=r"\Users\Jzimi\git-repos\forks\spikeinterface\playing",
	)
	# Automatic AP preprocessing preserves the raw AP channel order and geometry.
	plot_recordings = {
		"raw_ap_recording": recording,
		"preprocessed_ap_recording": recording,
		"raw_lfp_recording": lfp_recording,
	}
	for segment_index in range(recording.get_num_segments()):
		plot_raw_data_quality_metrics(
			results["chunked_results"], results["rolling_rms"],
			segment_index=segment_index, recordings=plot_recordings,
		)
	plt.show()
