"""
Full pipeline: preprocessing, spike sorting, and bombcell quality control.

Neuropixels analysis pipeline: load SpikeGLX recording, preprocess,
run Kilosort4, compute quality/template metrics, and run bombcell to
label units as good, MUA, noise, or non-somatic.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import spikeinterface.full as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

# %% Paths — edit these

spikeglx_folder = Path("/path/to/your/spikeglx/recording")
base_folder = spikeglx_folder

preprocessed_folder = base_folder / "preprocessed"
kilosort_folder = base_folder / "kilosort4_output"
analyzer_folder = base_folder / "sorting_analyzer.zarr"

preprocessed_exists = (preprocessed_folder / "si_folder.json").exists()

job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)

# %% 1. Load recording

raw_rec = si.read_spikeglx(spikeglx_folder, stream_name="imec0.ap", load_sync_channel=False)
print(raw_rec)

# %% 2. Preprocess
# Highpass → bad channel removal → phase shift → common median reference.
# All lazy until saved.

if not preprocessed_exists:
    rec_filtered = si.highpass_filter(raw_rec, freq_min=300.0)

    bad_channel_ids, channel_labels = si.detect_bad_channels(rec_filtered)
    print(f"Bad channels detected: {bad_channel_ids}")
    rec_clean = rec_filtered.remove_channels(bad_channel_ids)

    # Save bad channel info
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    with open(preprocessed_folder / "bad_channels.json", "w") as f:
        json.dump({"bad_channel_ids": [str(ch) for ch in bad_channel_ids]}, f, indent=2)

    rec_shifted = si.phase_shift(rec_clean)
    rec_cmr = si.common_reference(rec_shifted, reference="global", operator="median")

    # Save to disk (Kilosort needs binary)
    rec_preprocessed = rec_cmr.save(folder=preprocessed_folder, format="binary", **job_kwargs)
else:
    print(f"Loading preprocessed recording from {preprocessed_folder}")
    rec_preprocessed = si.load(preprocessed_folder)

print(rec_preprocessed)

# %% 3. Run Kilosort4

if kilosort_folder.exists():
    print(f"Loading existing Kilosort4 output from {kilosort_folder}")
    # register_recording=False: avoids errors when the original recording
    # path no longer exists (e.g. different mount point)
    sorting = si.read_sorter_folder(kilosort_folder, register_recording=False)
else:
    sorting = si.run_sorter(
        sorter_name="kilosort4",
        recording=rec_preprocessed,
        folder=kilosort_folder,
        remove_existing_folder=True,
        verbose=True,
        skip_kilosort_preprocessing=True,
        do_CAR=False,
    )
print(f"Kilosort4 found {len(sorting.unit_ids)} units")

# %% 4. Create SortingAnalyzer and compute extensions

if analyzer_folder.exists():
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

# Core extensions
if not analyzer.has_extension("random_spikes"):
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
if not analyzer.has_extension("waveforms"):
    analyzer.compute("waveforms", ms_before=3.0, ms_after=3.0, **job_kwargs)
if not analyzer.has_extension("templates"):
    analyzer.compute("templates", operators=["average", "median", "std"])
if not analyzer.has_extension("noise_levels"):
    analyzer.compute("noise_levels")

# Quality metric prerequisites
if not analyzer.has_extension("spike_amplitudes"):
    analyzer.compute("spike_amplitudes", **job_kwargs)
if not analyzer.has_extension("unit_locations"):
    analyzer.compute("unit_locations")
if not analyzer.has_extension("spike_locations"):
    analyzer.compute("spike_locations", **job_kwargs)

# Template metrics (include_multi_channel_metrics for exp_decay)
if not analyzer.has_extension("template_metrics"):
    analyzer.compute("template_metrics", include_multi_channel_metrics=True)

# %% 5. Configure and compute quality metrics

# Toggle options
compute_distance_metrics = False  # needs PCA; best for stable/chronic recordings
compute_drift = True
label_non_somatic = True
split_non_somatic_good_mua = False
use_valid_periods = False  # compute quality metrics only on good time chunks

# RPV method: "sliding_rp" (default, sweeps RP range) or "llobet" (single RP value)
rp_violation_method = "sliding_rp"

refractory_period_ms = 2.0
censored_period_ms = 0.1

qm_params = {
    "presence_ratio": {"bin_duration_s": 60},
    "rp_violation": {"refractory_period_ms": refractory_period_ms, "censored_period_ms": censored_period_ms},
    "sliding_rp_violation": {
        "exclude_ref_period_below_ms": 0.5,
        "max_ref_period_ms": 10.0,
        "confidence_threshold": 0.9,
    },
    "drift": {"interval_s": 60, "min_spikes_per_interval": 100},
}

# Valid time periods parameters (only used if use_valid_periods = True)
# fp_threshold and fn_threshold are auto-derived from bombcell thresholds
valid_periods_params = {
    "refractory_period_ms": refractory_period_ms,
    "censored_period_ms": censored_period_ms,
    "period_mode": "absolute",
    "period_duration_s_absolute": 30.0,
    "minimum_valid_period_duration": 180,
}

metric_names = ["amplitude_median", "snr", "amplitude_cutoff", "num_spikes", "presence_ratio", "firing_rate"]

if rp_violation_method == "sliding_rp":
    metric_names.append("sliding_rp_violation")
else:
    metric_names.append("rp_violation")

if compute_drift:
    metric_names.append("drift")

if compute_distance_metrics:
    metric_names.append("mahalanobis")  # produces isolation_distance and l_ratio
    if not analyzer.has_extension("principal_components"):
        analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)

if use_valid_periods and not analyzer.has_extension("amplitude_scalings"):
    analyzer.compute("amplitude_scalings", **job_kwargs)

if analyzer.has_extension("quality_metrics"):
    analyzer.delete_extension("quality_metrics")
analyzer.compute("quality_metrics", metric_names=metric_names, metric_params=qm_params, **job_kwargs)

# %% 6. Run bombcell
#
# The thresholds dict has three sections: "noise", "mua", "non-somatic".
# Each entry is {"greater": val, "less": val} (use None to disable one side).
#
# You can add any metric from the analyzer's DataFrame to any section.
# Custom metrics in "non-somatic" are OR'd with the built-in waveform shape logic.
# Metrics that haven't been computed are skipped with a warning.

thresholds = sc.bombcell_get_default_thresholds()

# Adjust existing thresholds
thresholds["mua"]["rpv"]["less"] = 0.1
thresholds["mua"]["presence_ratio"]["greater"] = 0.7

# Add custom metrics — uncomment any of these:
# thresholds["mua"]["firing_rate"] = {"greater": 0.1, "less": None}
# thresholds["mua"]["silhouette"] = {"greater": 0.4, "less": None}
# thresholds["noise"]["half_width"] = {"greater": 0.05e-3, "less": 0.6e-3}
# thresholds["non-somatic"]["velocity"] = {"greater": 2.0, "less": None}

# Disable a threshold:
# thresholds["mua"]["drift_ptp"] = {"greater": None, "less": None}

pprint(thresholds)

bombcell_labels = sc.bombcell_label_units(
    sorting_analyzer=analyzer,
    thresholds=thresholds,
    label_non_somatic=label_non_somatic,
    split_non_somatic_good_mua=split_non_somatic_good_mua,
    use_valid_periods=use_valid_periods,
    valid_periods_params=valid_periods_params if use_valid_periods else None,
    **job_kwargs,
)

print(f"\nLabeled {len(bombcell_labels)} units")
print(bombcell_labels["bombcell_label"].value_counts())

# %% 7. Visualize

sw.plot_unit_labels(analyzer, bombcell_labels["bombcell_label"], ylims=(-300, 100))
sw.plot_metric_histograms(analyzer, thresholds, figsize=(15, 10))
sw.plot_bombcell_labels_upset(
    analyzer,
    unit_labels=bombcell_labels["bombcell_label"],
    thresholds=thresholds,
    unit_labels_to_plot=["noise", "mua"],  # add "non_soma" to see non-somatic patterns
)
plt.show()

# %% 8. Remove noise units

analyzer_clean_folder = base_folder / "sorting_analyzer_clean.zarr"

if analyzer_clean_folder.exists():
    analyzer_clean = si.load_sorting_analyzer(analyzer_clean_folder)
else:
    non_noise = bombcell_labels["bombcell_label"] != "noise"
    analyzer_clean = analyzer.select_units(
        analyzer.unit_ids[non_noise],
        folder=analyzer_clean_folder,
        format="zarr",
    )
print(f"Kept {len(analyzer_clean.unit_ids)} / {len(analyzer.unit_ids)} units after removing noise")

# %% Notes on parameter tuning by recording type
#
# Chronic: set compute_distance_metrics=True, increase/disable drift threshold
# Acute: keep compute_distance_metrics=False, keep drift threshold strict
# Cerebellum: relax num_positive_peaks (complex spikes), shorter peak_to_trough_duration
# Striatum: lower spike count and presence ratio thresholds for MSNs
