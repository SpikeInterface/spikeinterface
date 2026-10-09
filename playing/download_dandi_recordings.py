#!/usr/bin/env python3
"""Stream DANDI NWB recordings and save using SpikeInterface.

Install:
    python -m pip install spikeinterface h5py remfile

Examples:
    python download_dandi_recordings.py
    python download_dandi_recordings.py --full --only ibl_np
    python download_dandi_recordings.py --full --only mouse_64ch

Outputs are SpikeInterface recording folders, not .nwb files.
By default, only the IBL five-minute cut is saved, beside this script in recordings/.
Full recordings require --full.
Existing completed outputs are skipped. Interrupted saves do not resume.
"""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import spikeinterface.extractors as se


API = "https://api.dandiarchive.org/api"

RECORDINGS = {
    "ibl_np": (
        "c9e5e89c-6365-4453-b426-84ff30d6f9b8",
        "acquisition/ElectricalSeriesProbe00AP",
    ),
    "mouse_64ch": (
        "80926e89-361f-4a4c-b3ce-d97b2ef2dcf4",
        "acquisition/extracellular array recording",
    ),
}


def open_recording(name):
    asset_id, series_path = RECORDINGS[name]

    with urlopen(f"{API}/assets/{asset_id}/", timeout=60) as response:
        metadata = json.load(response)

    url = next(
        (
            url
            for url in metadata.get("contentUrl", [])
            if "s3" in url
        ),
        f"{API}/assets/{asset_id}/download/",
    )

    print(f"Opening {name}: {metadata['path']}", flush=True)

    return se.read_nwb_recording(
        file_path=url,
        electrical_series_path=series_path,
        stream_mode="remfile",
        load_channel_properties=True,
        # Preserve exact timestamps. This loads the time vector into RAM,
        # but the much larger trace array remains streamed.
        load_time_vector=True,
    )


def save_recording(recording, folder):
    marker = folder / "download_complete.json"

    if marker.exists():
        print(f"Already saved: {folder}", flush=True)
        return

    if folder.exists():
        raise RuntimeError(
            f"{folder} exists without a completion marker. "
            "Move or remove that incomplete folder before rerunning."
        )

    print(f"Saving {folder}", flush=True)

    recording.save(
        folder=folder,
        format="binary",
        n_jobs=14,
        chunk_duration="1s",
        progress_bar=True,
    )

    marker.write_text(
        json.dumps(
            {
                "num_channels": recording.get_num_channels(),
                "num_samples": recording.get_num_samples(),
                "sampling_frequency_hz": recording.get_sampling_frequency(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "recordings")
    parser.add_argument("--only", nargs="+", choices=list(RECORDINGS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Also save full-length recordings.")
    mode.add_argument(
        "--short-only",
        action="store_true",
        help="Save only the IBL five-minute cut (the default).",
    )
    args = parser.parse_args()

    selected = args.only or (list(RECORDINGS) if args.full else ["ibl_np"])

    if not args.full and selected != ["ibl_np"]:
        parser.error("Use --full to download recordings other than the IBL five-minute cut")

    args.output.mkdir(parents=True, exist_ok=True)
    ibl = None

    # Save the quick-test recording before any full trace transfer.
    if "ibl_np" in selected:
        folder = args.output / "ibl_np_5min"

        if not (folder / "download_complete.json").exists():
            ibl = open_recording("ibl_np")
            times = ibl.get_times()

            if times[-1] < times[0] + 300:
                raise RuntimeError("IBL recording is shorter than five minutes")

            end_frame = int(np.searchsorted(times, times[0] + 300))
            cut = ibl.frame_slice(start_frame=0, end_frame=end_frame)
            save_recording(cut, folder)
        else:
            print(f"Already saved: {folder}", flush=True)

    if not args.full:
        return

    for name in selected:
        folder = args.output / name

        if (folder / "download_complete.json").exists():
            print(f"Already saved: {folder}", flush=True)
            continue

        recording = (
            ibl
            if name == "ibl_np" and ibl is not None
            else open_recording(name)
        )
        save_recording(recording, folder)


if __name__ == "__main__":
    main()
