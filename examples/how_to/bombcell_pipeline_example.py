"""
BombCell pipeline example.

This example shows how to use the preprocessing and QC wrapper functions
with customizable parameters and thresholds.
"""

from pathlib import Path
import spikeinterface.full as si
import spikeinterface.curation as sc

# %% Paths - edit these to match your data
spikeglx_folder = Path("/path/to/your/spikeglx/recording")  # folder containing .ap.bin and .ap.meta files
sorting_folder = spikeglx_folder / "kilosort4_output"       # folder with spike sorter output (spike_times.npy, etc.)
analyzer_folder = spikeglx_folder / "sorting_analyzer.zarr" # where to save the SortingAnalyzer (will be created)
output_folder = spikeglx_folder / "bombcell"                # where to save BombCell results (metrics, plots, labels)

# %% Load data
# NOTE: Recording and sorting are always needed - even when loading an existing analyzer,
# because the analyzer may need the recording for some computations.
recording = si.read_spikeglx(spikeglx_folder, stream_name="imec0.ap", load_sync_channel=False)  # load raw recording
sorting = si.read_sorter_folder(sorting_folder, register_recording=False)                       # load spike sorting results

# %% Preprocessing parameters
# Get defaults and modify as needed
preproc_params = sc.get_default_preprocessing_params()

# Example parameters you may want to modify:

# --- Filtering ---
preproc_params["freq_min"] = 300.0                      # highpass cutoff in Hz - removes LFP/low-frequency noise

# --- Bad channel detection ---
preproc_params["detect_bad_channels"] = True            # auto-detect and remove bad/dead channels
preproc_params["bad_channel_method"] = "coherence+psd"  # "coherence+psd" (recommended), "mad", or "std"

# --- Phase shift (essential for Neuropixels) ---
preproc_params["apply_phase_shift"] = True              # correct small timing offsets from multiplexed ADC sampling
                                                        # critical for common reference to work properly

# --- Common reference ---
preproc_params["apply_cmr"] = True                      # subtract common signal to remove shared noise
preproc_params["cmr_reference"] = "global"              # "global" (all chans), "local" (nearby), "single" (one chan)
preproc_params["cmr_operator"] = "median"               # "median" (robust to outliers) or "average"

# --- Waveform extraction ---
preproc_params["max_spikes_per_unit"] = 500             # spikes to extract per unit (more = accurate but slower)
preproc_params["ms_before"] = 3.0                       # ms before spike peak to include in waveform
preproc_params["ms_after"] = 3.0                        # ms after spike peak to include in waveform 

# %% QC parameters
qc_params = sc.get_default_qc_params()

# Example parameters you may want to modify:

# --- Metrics to compute ---
qc_params["compute_drift"] = True                       # compute drift metrics (position changes over time)
qc_params["compute_distance_metrics"] = False           # isolation distance & L-ratio - slow, not drift-robust
                                                        # recommend True for stable/chronic recordings
qc_params["rp_method"] = "sliding_rp"                   # refractory period method: "sliding_rp" or "llobet"

# --- BombCell classification options ---
qc_params["label_non_somatic"] = True                   # detect axonal/dendritic units via waveform shape
qc_params["split_non_somatic_good_mua"] = False         # if True, non-somatic split into good/mua subcategories
qc_params["use_valid_periods"] = False                  # if True, identify valid time chunks per unit and recompute
                                                        # quality metrics only on those periods (see below for details)

# --- Presence ratio ---
qc_params["presence_ratio_bin_duration_s"] = 60         # bin size (s) for checking if unit fires throughout recording

# --- Refractory period violations ---
qc_params["refractory_period_ms"] = 2.0                 # expected refractory period - use 1.0-1.5 for fast-spiking
qc_params["censored_period_ms"] = 0.1                   # ignore ISIs shorter than this (spike sorting artifact)

# --- Sliding RP method parameters ---
qc_params["sliding_rp_exclude_below_ms"] = 0.5          # exclude ISIs below this when fitting contamination
qc_params["sliding_rp_max_ms"] = 10.0                   # max ISI to consider for refractory period analysis
qc_params["sliding_rp_confidence"] = 0.9                # confidence level for contamination estimate (0-1)

# --- Drift parameters ---
qc_params["drift_interval_s"] = 60                      # time bin (s) for computing position over time
qc_params["drift_min_spikes"] = 100                     # min spikes in bin to estimate position (skip if fewer)

# --- Plotting ---
qc_params["plot_histograms"] = True                     # save histogram plots of all metrics
qc_params["plot_waveforms"] = True                      # save waveform plots for each unit
qc_params["plot_upset"] = True                          # save UpSet plot showing threshold failure combinations

# %% Classification thresholds
# Format: {"greater": min_value, "less": max_value} - unit passes if min < value < max
# Use None to disable a bound. Add "abs": True to use absolute value.
thresholds = sc.bombcell_get_default_thresholds()

# --- Noise thresholds (waveform quality) ---
# Units failing ANY of these are labeled "noise" (not neural signals)
thresholds["noise"]["num_positive_peaks"] = {"greater": None, "less": 2}             # max positive peaks in waveform (>1 = multi-unit/noise)
thresholds["noise"]["num_negative_peaks"] = {"greater": None, "less": 1}             # max negative peaks (>0 unusual for somatic spikes)
thresholds["noise"]["peak_to_trough_duration"] = {"greater": 0.0001, "less": 0.00115}  # spike width in seconds (0.1-1.15ms is physiological)
thresholds["noise"]["waveform_baseline_flatness"] = {"greater": None, "less": 0.5}   # baseline variation (high = noisy/unstable)
thresholds["noise"]["peak_after_to_trough_ratio"] = {"greater": None, "less": 0.8}   # repolarization peak vs trough amplitude
thresholds["noise"]["exp_decay"] = {"greater": 0.01, "less": 0.1}                    # exponential decay constant of waveform tail

# --- MUA thresholds (spike quality) ---
# Units failing ANY of these (that passed noise) are labeled "mua" (multi-unit activity)
thresholds["mua"]["amplitude_median"] = {"greater": 30, "less": None, "abs": True}   # minimum amplitude in uV (abs=True uses |amplitude|)
thresholds["mua"]["snr"] = {"greater": 5, "less": None}                              # signal-to-noise ratio (higher = cleaner unit)
thresholds["mua"]["amplitude_cutoff"] = {"greater": None, "less": 0.2}               # fraction of spikes below detection threshold (0 = none missing)
thresholds["mua"]["num_spikes"] = {"greater": 300, "less": None}                     # minimum spike count (too few = unreliable metrics)
thresholds["mua"]["rpv"] = {"greater": None, "less": 0.1}                            # refractory period violation rate (0 = perfect isolation)
thresholds["mua"]["presence_ratio"] = {"greater": 0.7, "less": None}                 # fraction of recording with spikes (1 = fires throughout)
thresholds["mua"]["drift_ptp"] = {"greater": None, "less": 100}                      # peak-to-peak position drift in um (lower = more stable)

# Optional distance metrics (only used if compute_distance_metrics=True)
thresholds["mua"]["isolation_distance"] = {"greater": 20, "less": None}              # Mahalanobis distance to nearest cluster (higher = better isolated)
thresholds["mua"]["l_ratio"] = {"greater": None, "less": 0.3}                        # L-ratio contamination estimate (lower = better isolated)

# --- Non-somatic thresholds (waveform shape) ---
# Detects axonal/dendritic units based on waveform features (these have different shapes than somatic spikes)
thresholds["non-somatic"]["peak_before_to_trough_ratio"] = {"greater": None, "less": 3}       # ratio of pre-peak to trough amplitude
thresholds["non-somatic"]["peak_before_width"] = {"greater": 0.00015, "less": None}           # width of peak before trough in seconds
thresholds["non-somatic"]["trough_width"] = {"greater": 0.0002, "less": None}                 # width of main trough in seconds
thresholds["non-somatic"]["peak_before_to_peak_after_ratio"] = {"greater": None, "less": 3}   # ratio of pre-peak to post-peak amplitude
thresholds["non-somatic"]["main_peak_to_trough_ratio"] = {"greater": None, "less": 0.8}       # ratio of main peak to trough amplitude

# %% Adding custom quality metrics
# You can add ANY metric from the SortingAnalyzer's quality_metrics or
# template_metrics DataFrame to ANY threshold section (noise, mua, non-somatic).
#
# How it works:
#   - Metrics in "noise" section: unit fails if ANY threshold is violated → labeled "noise"
#   - Metrics in "mua" section: unit fails if ANY threshold is violated → labeled "mua"
#   - Metrics in "non-somatic" section: OR'd with built-in waveform shape checks
#   - Metrics that haven't been computed are automatically skipped (with a warning)
#
# Threshold format:
#   {"greater": min_value, "less": max_value}  - unit passes if min < value < max
#   {"greater": min_value, "less": max_value, "abs": True}  - uses |value| for comparison
#   Use None to disable one bound (e.g., {"greater": 0.1, "less": None} means value > 0.1)
#
# Examples of adding custom metrics:
# thresholds["mua"]["firing_rate"] = {"greater": 0.1, "less": None}           # exclude units with firing rate < 0.1 Hz
# thresholds["mua"]["silhouette"] = {"greater": 0.4, "less": None}            # silhouette score (requires PCA)
# thresholds["noise"]["half_width"] = {"greater": 0.05e-3, "less": 0.6e-3}    # spike half-width bounds (template_metrics)
# thresholds["non-somatic"]["velocity_above"] = {"greater": 2.0, "less": None}  # axonal propagation velocity
#
# To DISABLE an existing threshold (skip it entirely):
# thresholds["mua"]["drift_ptp"] = {"greater": None, "less": None}            # both bounds None = threshold ignored
#
# Available metrics depend on what extensions are computed. Common ones include:
#   Quality metrics: amplitude_median, snr, amplitude_cutoff, num_spikes, presence_ratio,
#                    firing_rate, isi_violation, sliding_rp_violation, drift, isolation_distance, l_ratio
#   Template metrics: peak_to_valley, half_width, repolarization_slope, recovery_slope,
#                     num_positive_peaks, num_negative_peaks, velocity_above, velocity_below, exp_decay

# %% Step 1: Preprocess and create analyzer
# This applies all preprocessing steps and extracts waveforms into a SortingAnalyzer.
#
# IMPORTANT: If analyzer_folder already exists, the existing analyzer is LOADED (not recreated).
# Extensions are only computed if they don't already exist - nothing is recomputed by default.
# To force recomputation, use rerun_extensions=True.
analyzer, rec_preprocessed, bad_channels = sc.preprocess_for_bombcell(
    recording=recording,              # raw recording object
    sorting=sorting,                  # spike sorting results
    analyzer_folder=analyzer_folder,  # if exists: loads it; if not: creates it
    params=preproc_params,            # preprocessing parameters defined above
    rerun_extensions=False,           # False (default): skip existing extensions; True: recompute all
    n_jobs=-1,                        # parallel jobs: -1 = all CPUs, 1 = single-threaded
)
# Returns:
#   analyzer: SortingAnalyzer with waveforms, templates, and extensions computed
#   rec_preprocessed: the preprocessed recording (filtered, referenced, etc.)
#   bad_channels: list of channel IDs that were detected as bad and removed (None if loaded from disk)
print(f"Bad channels removed: {bad_channels}")

# %% Step 2: Run BombCell QC
# This computes quality metrics and classifies units as good/mua/noise/non-somatic.
#
# IMPORTANT: Quality metrics are only computed if they don't already exist in the analyzer.
# To force recomputation (e.g., after changing qc_params), use rerun_quality_metrics=True.
labels, metrics, figures = sc.run_bombcell_qc(
    sorting_analyzer=analyzer,        # SortingAnalyzer from step 1
    output_folder=output_folder,      # where to save results (CSVs, plots)
    params=qc_params,                 # QC parameters defined above
    thresholds=thresholds,            # classification thresholds defined above
    rerun_quality_metrics=False,      # False (default): use existing metrics; True: recompute
    n_jobs=-1,                        # parallel jobs: -1 = all CPUs, 1 = single-threaded
)
# Returns:
#   labels: DataFrame with unit_id index and 'bombcell_label' column (good/mua/noise/non_soma)
#   metrics: DataFrame with all computed quality metrics for each unit
#   figures: dict of matplotlib figures (histograms, waveforms, upset plot)

# %% Results
print(f"\nResults saved to: {output_folder}")
print(f"\nLabel distribution:\n{labels['bombcell_label'].value_counts()}")

# Get units by label
good_units = labels[labels["bombcell_label"] == "good"].index.tolist()
mua_units = labels[labels["bombcell_label"] == "mua"].index.tolist()
noise_units = labels[labels["bombcell_label"] == "noise"].index.tolist()
non_soma_units = labels[labels["bombcell_label"] == "non_soma"].index.tolist()

print(f"\nGood units ({len(good_units)}): {good_units[:10]}...")
print(f"MUA units ({len(mua_units)}): {mua_units[:10]}...")

# %% Access metrics for specific units
print(f"\nMetrics for first good unit:")
if good_units:
    print(metrics.loc[good_units[0]])

# %% Output files
# BombCell saves the following files to output_folder:
#
# labeling_results_wide.csv
#   - One row per unit, all metrics as columns, plus "label" column
#   - Format: unit_id (index), label, metric1, metric2, ...
#   - Use for quick overview of all units and their metrics
#
# labeling_results_narrow.csv
#   - One row per unit-metric combination (tidy/long format)
#   - Columns: unit_id, label, metric_name, value, threshold_min, threshold_max, passed
#   - Use to see exactly which metrics failed for each unit
#
# valid_periods.tsv (only if use_valid_periods=True)
#   - Valid time periods per unit for downstream analysis
#   - Columns: unit_id, segment_index, start_time_s, end_time_s, duration_s
#   - Use to filter spikes to stable periods in your analysis
#
# metric_histograms.png
#   - Histogram of each metric with threshold lines marked
#   - Useful for adjusting thresholds based on your data distribution
#
# waveforms_by_label.png
#   - Waveform overlays grouped by label (good, mua, noise, non_soma)
#   - Verify that labels match expected waveform shapes
#
# upset_plot_*.png
#   - UpSet plots showing which metrics fail together
#   - Understand why units are labeled noise/mua

# %% Using valid time periods
# Valid periods identify chunks of time where each unit has stable amplitude
# and low refractory period violations. This is useful when recordings have
# unstable periods (e.g., drift, probe movement, or electrode noise).
#
# When use_valid_periods=True:
#   1. Recording is divided into chunks (default 30s or ~300 spikes per unit)
#   2. For each chunk, false positive rate (RP violations) and false negative
#      rate (amplitude cutoff) are computed
#   3. Chunks where BOTH rates are below threshold are marked as "valid"
#   4. Overlapping valid chunks are merged; short periods (<180s) are removed
#   5. Quality metrics are recomputed using only spikes within valid periods
#   6. BombCell labeling is applied to these restricted metrics
#   7. valid_periods.tsv is saved with the valid time windows per unit
#
# Example: Enable valid periods
# qc_params["use_valid_periods"] = True
#
# Example: Customize valid period parameters
# valid_periods_params = {
#     "period_duration_s_absolute": 30.0,      # chunk size in seconds (if period_mode="absolute")
#     "period_target_num_spikes": 300,         # target spikes per chunk (if period_mode="relative")
#     "period_mode": "absolute",               # "absolute" (fixed duration) or "relative" (fixed spike count)
#     "minimum_valid_period_duration": 180,    # min duration to keep a valid period (seconds)
#     "fp_threshold": 0.1,                     # max false positive rate (derived from rpv threshold if not set)
#     "fn_threshold": 0.1,                     # max false negative rate (derived from amplitude_cutoff if not set)
# }
# labels, metrics, figures = sc.run_bombcell_qc(
#     analyzer, params=qc_params, valid_periods_params=valid_periods_params
# )
#
# Example: Load valid_periods.tsv for downstream analysis
# import pandas as pd
# valid_periods = pd.read_csv(output_folder / "valid_periods.tsv", sep="\t")
# # Filter to get valid periods for a specific unit
# unit_periods = valid_periods[valid_periods["unit_id"] == good_units[0]]
# print(f"Unit {good_units[0]} has {len(unit_periods)} valid period(s)")
