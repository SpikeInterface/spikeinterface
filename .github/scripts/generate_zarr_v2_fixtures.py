#!/usr/bin/env python3
"""
Generate zarr v2 fixtures for backward compatibility tests.

Run this script with an old spikeinterface version and zarr<3, e.g.:
    pip install "spikeinterface==0.104.0" "zarr<3"
    python generate_zarr_v2_fixtures.py --output /tmp/zarr_v2_fixtures

The script saves:
  - recording.zarr  : a small ZarrRecordingExtractor
  - sorting.zarr    : a small ZarrSortingExtractor
  - expected_values.json : key values used to verify correct loading
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(output_dir: Path) -> None:
    import spikeinterface

    print(f"spikeinterface version : {spikeinterface.__version__}")

    import zarr

    print(f"zarr version           : {zarr.__version__}")

    from spikeinterface.core import generate_recording, generate_sorting
    from spikeinterface.core import ZarrRecordingExtractor, ZarrSortingExtractor
    from spikeinterface.core import create_sorting_analyzer, load_sorting_analyzer

    output_dir.mkdir(parents=True, exist_ok=True)

    recording = generate_recording(num_channels=4, num_segments=2, seed=0)
    sorting = generate_sorting(num_units=3, num_segments=2, seed=0)

    # --- save recording ---
    recording_path = output_dir / "recording.zarr"
    ZarrRecordingExtractor.write_recording(recording, recording_path)
    print(f"Saved recording  -> {recording_path}")

    # --- save sorting ---
    sorting_path = output_dir / "sorting.zarr"
    ZarrSortingExtractor.write_sorting(sorting, sorting_path)
    print(f"Saved sorting    -> {sorting_path}")

    # --- save SortingAnalyzer ---
    # Reload the recording from zarr so it is a serializable ZarrRecordingExtractor,
    # which the analyzer can store as provenance.
    recording_zarr = ZarrRecordingExtractor(recording_path)
    analyzer_path = output_dir / "analyzer.zarr"
    analyzer = create_sorting_analyzer(
        sorting, recording_zarr, format="zarr", folder=analyzer_path, sparse=False, sparsity=None
    )
    analyzer.compute(["random_spikes", "templates"])
    print(f"Saved analyzer   -> {analyzer_path}")

    # Reload to verify templates are accessible before writing expected values
    analyzer = load_sorting_analyzer(analyzer_path)
    templates_array = analyzer.get_extension("templates").get_data()

    # --- capture expected values for later assertion ---
    expected = {
        "spikeinterface_version": spikeinterface.__version__,
        "zarr_version": zarr.__version__,
        "recording": {
            "num_channels": int(recording.get_num_channels()),
            "num_segments": int(recording.get_num_segments()),
            "sampling_frequency": float(recording.get_sampling_frequency()),
            "num_frames_per_segment": [int(recording.get_num_frames(seg)) for seg in range(recording.get_num_segments())],
            "channel_ids": recording.get_channel_ids().tolist(),
            "dtype": str(recording.get_dtype()),
            # first 10 frames of segment 0 for all channels
            "traces_seg0_first10": recording.get_traces(start_frame=0, end_frame=10, segment_index=0).tolist(),
        },
        "sorting": {
            "num_segments": int(sorting.get_num_segments()),
            "sampling_frequency": float(sorting.get_sampling_frequency()),
            "unit_ids": sorting.get_unit_ids().tolist(),
            "spike_trains_seg0": {
                str(uid): sorting.get_unit_spike_train(unit_id=uid, segment_index=0).tolist()
                for uid in sorting.unit_ids
            },
        },
        "analyzer": {
            "num_units": int(analyzer.get_num_units()),
            "num_channels": int(analyzer.get_num_channels()),
            "templates_shape": list(templates_array.shape),
        },
    }

    expected_path = output_dir / "expected_values.json"
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Saved expected   -> {expected_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate zarr v2 fixtures for backward compatibility tests")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write fixtures into")
    args = parser.parse_args()
    main(args.output)
