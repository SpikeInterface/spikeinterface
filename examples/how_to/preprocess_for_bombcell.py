"""
Preprocess recording and create SortingAnalyzer for BombCell.
Companion script: run_bombcell_labeling.py
"""

import json
from pathlib import Path
import spikeinterface.full as si

# %% Paths
recording_folder = Path("/path/to/your/recording")
preprocessed_folder = recording_folder / "preprocessed"
sorting_folder = recording_folder / "kilosort4_output"
analyzer_folder = recording_folder / "sorting_analyzer.zarr"

# %% Rerun flags (set True to force recompute)
rerun_preprocessing = False
rerun_sorting = False
rerun_analyzer = False
rerun_extensions = False

# %% Preprocessing parameters
preprocessing_params = dict(
    highpass_freq_min=300.0,
    detect_bad_channels=True,
    apply_phase_shift=True,
    apply_common_reference=True,
    cmr_reference="global",
    cmr_operator="median",
)

# %% Sorter parameters
sorter_name = "kilosort4"
sorter_params = dict(
    skip_kilosort_preprocessing=True,
    do_CAR=False,
)

# %% Extension parameters
extension_params = dict(
    random_spikes=dict(method="uniform", max_spikes_per_unit=500),
    waveforms=dict(ms_before=3.0, ms_after=3.0),
    templates=dict(operators=["average", "median", "std"]),
    template_metrics=dict(include_multi_channel_metrics=True),
)

job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)

# %% 1. Load recording
raw_rec = si.read_spikeglx(recording_folder, stream_name="imec0.ap", load_sync_channel=False)
print(f"Loaded: {raw_rec.get_num_channels()} channels, {raw_rec.get_total_duration():.1f}s")

# %% 2. Preprocess
if (preprocessed_folder / "si_folder.json").exists() and not rerun_preprocessing:
    print(f"Loading preprocessed from {preprocessed_folder}")
    rec_preprocessed = si.load(preprocessed_folder)
else:
    pp = preprocessing_params
    rec = si.highpass_filter(raw_rec, freq_min=pp["highpass_freq_min"])

    if pp["detect_bad_channels"]:
        bad_ids, labels = si.detect_bad_channels(rec)
        print(f"Bad channels: {list(bad_ids)}")
        rec = rec.remove_channels(bad_ids)
        preprocessed_folder.mkdir(parents=True, exist_ok=True)
        with open(preprocessed_folder / "bad_channels.json", "w") as f:
            json.dump({"bad_channel_ids": [str(c) for c in bad_ids]}, f)

    if pp["apply_phase_shift"]:
        rec = si.phase_shift(rec)
    if pp["apply_common_reference"]:
        rec = si.common_reference(rec, reference=pp["cmr_reference"], operator=pp["cmr_operator"])

    rec_preprocessed = rec.save(folder=preprocessed_folder, format="binary", **job_kwargs)

# %% 3. Spike sorting
if sorting_folder.exists() and not rerun_sorting:
    print(f"Loading sorting from {sorting_folder}")
    sorting = si.read_sorter_folder(sorting_folder, register_recording=False)
else:
    sorting = si.run_sorter(
        sorter_name=sorter_name,
        recording=rec_preprocessed,
        folder=sorting_folder,
        remove_existing_folder=True,
        verbose=True,
        **sorter_params,
    )
print(f"Units: {len(sorting.unit_ids)}")

# %% 4. Create SortingAnalyzer
if analyzer_folder.exists() and not rerun_analyzer:
    print(f"Loading analyzer from {analyzer_folder}")
    analyzer = si.load_sorting_analyzer(analyzer_folder)
    if not analyzer.has_recording():
        analyzer.set_temporary_recording(rec_preprocessed)
else:
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec_preprocessed,
        sparse=True,
        format="zarr",
        folder=analyzer_folder,
        return_in_uV=True,
    )

# %% 5. Compute extensions
def compute_ext(name, **kwargs):
    if analyzer.has_extension(name) and not rerun_extensions:
        return
    if analyzer.has_extension(name):
        analyzer.delete_extension(name)
    print(f"Computing {name}...")
    analyzer.compute(name, **kwargs)

compute_ext("random_spikes", **extension_params["random_spikes"])
compute_ext("waveforms", **extension_params["waveforms"], **job_kwargs)
compute_ext("templates", **extension_params["templates"])
compute_ext("noise_levels")
compute_ext("spike_amplitudes", **job_kwargs)
compute_ext("unit_locations")
compute_ext("spike_locations", **job_kwargs)
compute_ext("template_metrics", **extension_params["template_metrics"])

print(f"\nDone. Analyzer saved to {analyzer_folder}")
print(f"Extensions: {analyzer.get_loaded_extension_names()}")
print(f"Next: run run_bombcell_labeling.py")
