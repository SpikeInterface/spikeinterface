"""
bombcell quality control example.

This example shows how to run bombcell quality control on a SortingAnalyzer
with customizable parameters and thresholds.

Prerequisites: a SortingAnalyzer with waveforms, templates, template_metrics,
and other extensions already computed. See SpikeInterface documentation for
preprocessing, spike sorting, and analyzer creation.
"""

# %%
import spikeinterface.full as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

from pathlib import Path
from pprint import pprint

# %% Paths - edit these to match your data
analyzer_folder = Path("/path/to/your/sorting_analyzer.zarr")
output_folder = Path("/path/to/your/bombcell_output")

# %% Load existing SortingAnalyzer
# The analyzer should already have: random_spikes, waveforms, templates,
# noise_levels, unit_locations, spike_locations, template_metrics.
# amplitude_scalings is computed on-demand if needed (for amplitude_cutoff or valid periods).
# See SpikeInterface docs for preprocessing, sorting, and analyzer creation.
analyzer = si.load_sorting_analyzer(analyzer_folder)

# %% QC parameters
# Get defaults and modify as needed
qc_params = sc.get_default_qc_params()

# --- Metrics to compute ---
qc_params["compute_amplitude_cutoff"] = True            # estimate missing spikes (requires spike_amplitudes)
qc_params["compute_drift"] = True                       # compute drift metrics (position changes over time)
qc_params["compute_distance_metrics"] = False            # isolation distance & L-ratio - slow, not drift-robust
                                                         # recommend True for stable/chronic recordings

# --- Refractory period method ---
qc_params["rp_method"] = {
    "method": "sliding_rp",                              # "sliding_rp" (recommended) or "llobet"
    "exclude_ref_period_below_ms": 0.5,                  # min RP to consider when sweeping
    "max_ref_period_ms": 10.0,                           # max RP to consider when sweeping
}

# --- bombcell classification options ---
qc_params["label_non_somatic"] = True                    # detect axonal/dendritic units via waveform shape
qc_params["split_non_somatic_good_mua"] = False          # if True, non-somatic split into good/mua subcategories
qc_params["use_valid_periods"] = False                   # if True, identify valid time chunks per unit and recompute
                                                         # quality metrics only on those periods

# --- Presence ratio ---
qc_params["presence_ratio_bin_duration_s"] = 60          # bin size (s) for checking if unit fires throughout recording

# --- Refractory period violations ---
qc_params["refractory_period_ms"] = 2.0                  # expected refractory period - use 1.0-1.5 for fast-spiking
qc_params["censored_period_ms"] = 0.1                    # ignore ISIs shorter than this (spike sorting artifact)

# --- Drift parameters ---
qc_params["drift_interval_s"] = 60                       # time bin (s) for computing position over time
qc_params["drift_min_spikes"] = 100                      # min spikes in bin to estimate position (skip if fewer)

# --- Plotting ---
qc_params["plot_histograms"] = True                      # save histogram plots of all metrics
qc_params["plot_waveforms"] = True                       # save waveform plots for each unit
qc_params["plot_upset"] = True                           # save UpSet plot showing threshold failure combinations

# --- Custom metric names / params (optional) ---
# To bypass the compute_* flags and specify exactly which metrics to compute,
# set qc_params["metric_names"] to a list. Any SpikeInterface quality metric works.
# qc_params["metric_names"] = [
#     "amplitude_median", "snr", "num_spikes", "presence_ratio", "firing_rate",
#     "sliding_rp_violation", "drift", "silhouette",
# ]
#
# To override metric-specific params, set qc_params["metric_params"]:
# qc_params["metric_params"] = {
#     "silhouette": {"method": "simplified"},
#     "drift": {"interval_s": 30},
# }

# %% Classification thresholds
# Format: {"greater": min_value, "less": max_value} - unit passes if min < value < max
# Use None to disable a bound. Add "abs": True to use absolute value.
thresholds = sc.bombcell_get_default_thresholds()

# --- Noise thresholds (waveform quality) ---
# Units failing ANY of these are labeled "noise" (not neural signals)
thresholds["noise"]["num_positive_peaks"] = {"greater": None, "less": 2}
thresholds["noise"]["num_negative_peaks"] = {"greater": None, "less": 1}
thresholds["noise"]["peak_to_trough_duration"] = {"greater": 0.0001, "less": 0.00115}
thresholds["noise"]["waveform_baseline_flatness"] = {"greater": None, "less": 0.5}
thresholds["noise"]["peak_after_to_trough_ratio"] = {"greater": None, "less": 0.8}
thresholds["noise"]["exp_decay"] = {"greater": 0.01, "less": 0.1}

# --- MUA thresholds (spike quality) ---
# Units failing ANY of these (that passed noise) are labeled "mua" (multi-unit activity)
thresholds["mua"]["amplitude_median"] = {"greater": 30, "less": None, "abs": True}
thresholds["mua"]["snr"] = {"greater": 5, "less": None}
thresholds["mua"]["amplitude_cutoff"] = {"greater": None, "less": 0.2}
thresholds["mua"]["num_spikes"] = {"greater": 300, "less": None}
thresholds["mua"]["rpv"] = {"greater": None, "less": 0.1}
thresholds["mua"]["presence_ratio"] = {"greater": 0.7, "less": None}
thresholds["mua"]["drift_ptp"] = {"greater": None, "less": 100}

# Optional distance metrics (only used if compute_distance_metrics=True)
# thresholds["mua"]["isolation_distance"] = {"greater": 20, "less": None}
# thresholds["mua"]["l_ratio"] = {"greater": None, "less": 0.3}

# --- Non-somatic thresholds (waveform shape) ---
# Detects axonal/dendritic units based on waveform features
thresholds["non-somatic"]["peak_before_to_trough_ratio"] = {"greater": None, "less": 3}
thresholds["non-somatic"]["peak_before_width"] = {"greater": 0.00015, "less": None}
thresholds["non-somatic"]["trough_width"] = {"greater": 0.0002, "less": None}
thresholds["non-somatic"]["peak_before_to_peak_after_ratio"] = {"greater": None, "less": 3}
thresholds["non-somatic"]["main_peak_to_trough_ratio"] = {"greater": None, "less": 0.8}

# %% Adding custom metrics
# You can add ANY metric from the SortingAnalyzer's quality_metrics or
# template_metrics DataFrame to ANY threshold section (noise, mua, non-somatic).
#
# Metrics in "noise" section: unit fails if ANY threshold is violated -> labeled "noise"
# Metrics in "mua" section: unit fails if ANY threshold is violated -> labeled "mua"
# Metrics in "non-somatic" section: OR'd with built-in waveform shape checks
# Metrics that haven't been computed are automatically skipped (with a warning)
#
# Examples:
# thresholds["mua"]["firing_rate"] = {"greater": 0.1, "less": None}
# thresholds["noise"]["half_width"] = {"greater": 0.05e-3, "less": 0.6e-3}
# thresholds["non-somatic"]["velocity_above"] = {"greater": 2.0, "less": None}
#
# To DISABLE an existing threshold:
# thresholds["mua"]["drift_ptp"] = {"greater": None, "less": None}

pprint(thresholds)

# %% Run bombcell QC
# This computes quality metrics and classifies units as good/mua/noise/non-somatic.
# Both `params` and `thresholds` also accept a path to a JSON file:
# e.g. params="qc_params.json", thresholds="thresholds.json".
# After each run, the thresholds and bombcell-specific config are saved to
# output_folder as thresholds.json and bombcell_config.json for reproducibility.
#
# Rerun flags force recomputation of specific extensions (all default False):
#   rerun_quality_metrics    - quality_metrics
#   rerun_pca                - principal_components (for distance metrics)
#   rerun_amplitude_scalings - amplitude_scalings (prerequisite for amplitude_cutoff and valid periods)
#   rerun_valid_periods      - valid_unit_periods (the "good times" extension)
labels, metrics, figures = sc.run_bombcell_qc(
    sorting_analyzer=analyzer,
    output_folder=output_folder,
    params=qc_params,
    thresholds=thresholds,
    rerun_quality_metrics=False,
    n_jobs=-1,
)

# %% Results
print(f"\nResults saved to: {output_folder}")
print(f"\nLabel distribution:\n{labels['bombcell_label'].value_counts()}")

good_units = labels[labels["bombcell_label"] == "good"].index.tolist()
mua_units = labels[labels["bombcell_label"] == "mua"].index.tolist()
noise_units = labels[labels["bombcell_label"] == "noise"].index.tolist()
non_soma_units = labels[labels["bombcell_label"] == "non_soma"].index.tolist()

print(f"\nGood units ({len(good_units)}): {good_units[:10]}...")
print(f"MUA units ({len(mua_units)}): {mua_units[:10]}...")

# %% Visualize: template peaks and troughs
_ = sw.plot_template_peak_trough(
    analyzer,
    unit_ids=analyzer.unit_ids[:8],
    n_channels_around=2,
    unit_labels=labels["bombcell_label"],
    figsize=(20, 12),
)

# %% Using valid time periods
# Valid periods identify chunks of time where each unit has stable amplitude
# and low refractory period violations. This is useful when recordings have
# unstable periods (e.g., drift, probe movement, or electrode noise).
#
# Example: Enable valid periods
# qc_params["use_valid_periods"] = True
#
# Example: Customize valid period parameters
# valid_periods_params = {
#     "period_duration_s_absolute": 30.0,
#     "period_target_num_spikes": 300,
#     "period_mode": "absolute",
#     "minimum_valid_period_duration": 180,
#     "fp_threshold": 0.1,
#     "fn_threshold": 0.1,
# }
# labels, metrics, figures = sc.run_bombcell_qc(
#     analyzer, params=qc_params, valid_periods_params=valid_periods_params
# )

# %% Using bombcell_label_units directly (without the pipeline)
# If you want more control, you can call bombcell_label_units directly.
# This skips quality metric computation, plotting, and saving — you handle those yourself.

# Basic usage: just pass the analyzer and thresholds
labels_direct = sc.bombcell_label_units(
    sorting_analyzer=analyzer,
    thresholds=thresholds,
)

# With external metrics (e.g. from a CSV or custom computation):
# import pandas as pd
# my_metrics = pd.read_csv("my_metrics.csv", index_col=0)
# labels_direct = sc.bombcell_label_units(
#     external_metrics=my_metrics,
#     thresholds=thresholds,
# )

# With valid periods and explicit quality_metric_params:
# When use_valid_periods=True, quality metrics are recomputed on valid time chunks.
# Pass quality_metric_params to control exactly which metrics and params are used,
# instead of relying on whatever was previously computed on the analyzer.
# labels_direct = sc.bombcell_label_units(
#     sorting_analyzer=analyzer,
#     thresholds=thresholds,
#     use_valid_periods=True,
#     valid_periods_params={
#         "period_duration_s_absolute": 30.0,
#         "minimum_valid_period_duration": 180,
#     },
#     quality_metric_params={
#         "metric_names": ["amplitude_median", "snr", "num_spikes", "sliding_rp_violation",
#                          "presence_ratio", "firing_rate", "amplitude_cutoff", "drift"],
#         "metric_params": {
#             "sliding_rp_violation": {"exclude_ref_period_below_ms": 0.5, "max_ref_period_ms": 10.0},
#             "presence_ratio": {"bin_duration_s": 60},
#             "drift": {"interval_s": 60, "min_spikes_per_interval": 100},
#         },
#     },
#     n_jobs=-1,
# )

# Thresholds can also be loaded from a JSON file:
# labels_direct = sc.bombcell_label_units(
#     sorting_analyzer=analyzer,
#     thresholds="my_thresholds.json",
# )

# %% Parameter tuning by recording type
#
# Chronic recordings:
#   - Distance metrics work well (stable recordings = reliable isolation_distance/l_ratio)
#   - Set compute_distance_metrics = True
#   - Increase or disable drift threshold
#
# Acute recordings:
#   - Distance metrics unreliable (drift artificially lowers isolation_distance/l_ratio)
#   - Keep compute_distance_metrics = False (default)
#   - Keep drift threshold strict
#
# Cerebellum:
#   - Purkinje cells: shorter peak-to-trough duration
#   - Complex spikes may trigger noise detection; relax num_positive_peaks
#
# Striatum:
#   - MSNs: lower spike count and presence ratio thresholds
