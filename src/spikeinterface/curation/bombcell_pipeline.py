"""
BombCell pipeline functions for preprocessing and quality control.

Functions
---------
preprocess_for_bombcell
    Preprocess recording and create SortingAnalyzer with required extensions.
run_bombcell_qc
    Compute quality metrics, run BombCell labeling, and generate plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def preprocess_for_bombcell(
    recording,
    sorting,
    analyzer_folder: str | Path,
    # Preprocessing
    freq_min: float = 300.0,
    detect_bad_channels: bool = True,
    bad_channel_method: str = "coherence+psd",
    apply_phase_shift: bool = True,
    apply_cmr: bool = True,
    cmr_reference: str = "global",
    cmr_operator: str = "median",
    # Analyzer
    sparse: bool = True,
    return_in_uV: bool = True,
    # Extensions
    max_spikes_per_unit: int = 500,
    ms_before: float = 3.0,
    ms_after: float = 3.0,
    template_operators: list[str] = ("average", "median", "std"),
    include_multi_channel_metrics: bool = True,
    # Rerun flags
    rerun_extensions: bool = False,
    # Job
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Preprocess recording and create SortingAnalyzer with extensions for BombCell.

    Parameters
    ----------
    recording : BaseRecording
        Raw recording to preprocess.
    sorting : BaseSorting
        Spike sorting result.
    analyzer_folder : str or Path
        Path to save the SortingAnalyzer (zarr format).

    Preprocessing Parameters
    ------------------------
    freq_min : float, default: 300.0
        Highpass filter cutoff frequency in Hz.
    detect_bad_channels : bool, default: True
        Detect and remove bad channels.
    bad_channel_method : str, default: "coherence+psd"
        Method for bad channel detection.
    apply_phase_shift : bool, default: True
        Apply phase shift correction (for Neuropixels).
    apply_cmr : bool, default: True
        Apply common median reference.
    cmr_reference : str, default: "global"
        Reference type: "global", "local", or "single".
    cmr_operator : str, default: "median"
        Operator: "median" or "average".

    Analyzer Parameters
    -------------------
    sparse : bool, default: True
        Use sparse waveform representation.
    return_in_uV : bool, default: True
        Return waveforms in microvolts.

    Extension Parameters
    --------------------
    max_spikes_per_unit : int, default: 500
        Number of spikes to extract per unit for waveforms.
    ms_before : float, default: 3.0
        Milliseconds before spike peak for waveform extraction.
    ms_after : float, default: 3.0
        Milliseconds after spike peak for waveform extraction.
    template_operators : list, default: ("average", "median", "std")
        Template statistics to compute.
    include_multi_channel_metrics : bool, default: True
        Include multi-channel template metrics (exp_decay, etc.).
    rerun_extensions : bool, default: False
        Force recomputation of existing extensions.

    Job Parameters
    --------------
    n_jobs : int, default: -1
        Number of parallel jobs (-1 for all CPUs).
    progress_bar : bool, default: True
        Show progress bars.

    Returns
    -------
    analyzer : SortingAnalyzer
        SortingAnalyzer with computed extensions.
    rec_preprocessed : BaseRecording
        Preprocessed recording (lazy, not saved).
    bad_channel_ids : list or None
        List of removed bad channel IDs, or None if detection disabled.
    """
    import spikeinterface.full as si

    analyzer_folder = Path(analyzer_folder)
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    # Preprocess
    rec = si.highpass_filter(recording, freq_min=freq_min)

    bad_channel_ids = None
    if detect_bad_channels:
        bad_channel_ids, _ = si.detect_bad_channels(rec, method=bad_channel_method)
        bad_channel_ids = list(bad_channel_ids)
        if len(bad_channel_ids) > 0:
            rec = rec.remove_channels(bad_channel_ids)

    if apply_phase_shift:
        rec = si.phase_shift(rec)

    if apply_cmr:
        rec = si.common_reference(rec, reference=cmr_reference, operator=cmr_operator)

    rec_preprocessed = rec

    # Create or load analyzer
    if analyzer_folder.exists():
        analyzer = si.load_sorting_analyzer(analyzer_folder)
        if not analyzer.has_recording():
            analyzer.set_temporary_recording(rec_preprocessed)
    else:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec_preprocessed,
            sparse=sparse,
            format="zarr",
            folder=analyzer_folder,
            return_in_uV=return_in_uV,
        )

    # Compute extensions
    def _compute(name, **kwargs):
        if analyzer.has_extension(name) and not rerun_extensions:
            return
        if analyzer.has_extension(name):
            analyzer.delete_extension(name)
        analyzer.compute(name, **kwargs)

    _compute("random_spikes", method="uniform", max_spikes_per_unit=max_spikes_per_unit)
    _compute("waveforms", ms_before=ms_before, ms_after=ms_after, **job_kwargs)
    _compute("templates", operators=list(template_operators))
    _compute("noise_levels")
    _compute("spike_amplitudes", **job_kwargs)
    _compute("unit_locations")
    _compute("spike_locations", **job_kwargs)
    _compute("template_metrics", include_multi_channel_metrics=include_multi_channel_metrics)

    return analyzer, rec_preprocessed, bad_channel_ids


def run_bombcell_qc(
    sorting_analyzer,
    output_folder: str | Path | None = None,
    # Quality metric options
    compute_distance_metrics: bool = False,
    compute_drift: bool = True,
    rp_method: str = "sliding_rp",
    # Quality metric parameters
    qm_params: dict | None = None,
    # BombCell options
    label_non_somatic: bool = True,
    split_non_somatic_good_mua: bool = False,
    use_valid_periods: bool = False,
    valid_periods_params: dict | None = None,
    # Thresholds (None = use defaults)
    thresholds: dict | None = None,
    # Plotting
    plot_histograms: bool = True,
    plot_waveforms: bool = True,
    plot_upset: bool = True,
    waveform_ylims: tuple | None = (-300, 100),
    figsize_histograms: tuple = (15, 10),
    # Rerun
    rerun_quality_metrics: bool = False,
    rerun_pca: bool = False,
    # Job
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Compute quality metrics and run BombCell unit labeling.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        Analyzer with template_metrics computed (from preprocess_for_bombcell).
    output_folder : str, Path, or None, default: None
        Folder to save results (CSV files and plots). If None, results not saved.

    Quality Metric Options
    ----------------------
    compute_distance_metrics : bool, default: False
        Compute isolation_distance and l_ratio (requires PCA, slow).
    compute_drift : bool, default: True
        Compute drift metrics.
    rp_method : str, default: "sliding_rp"
        Refractory period violation method: "sliding_rp" or "llobet".

    Quality Metric Parameters
    -------------------------
    qm_params : dict or None, default: None
        Override default quality metric parameters. Keys are metric names,
        values are parameter dicts. Default parameters:

        - presence_ratio: {"bin_duration_s": 60}
        - rp_violation: {"refractory_period_ms": 2.0, "censored_period_ms": 0.1}
        - sliding_rp_violation: {"exclude_ref_period_below_ms": 0.5,
                                  "max_ref_period_ms": 10.0, "confidence_threshold": 0.9}
        - drift: {"interval_s": 60, "min_spikes_per_interval": 100}

    BombCell Options
    ----------------
    label_non_somatic : bool, default: True
        Detect non-somatic (axonal/dendritic) units.
    split_non_somatic_good_mua : bool, default: False
        Split non-somatic into "non_soma_good" and "non_soma_mua".
    use_valid_periods : bool, default: False
        Restrict metrics to valid time periods per unit.
    valid_periods_params : dict or None, default: None
        Parameters for valid_unit_periods extension.

    Classification Thresholds
    -------------------------
    thresholds : dict or None, default: None
        BombCell thresholds dict with "noise", "mua", "non-somatic" sections.
        If None, uses bombcell_get_default_thresholds(). Default values:

        noise (waveform quality - any failure -> "noise"):
            - num_positive_peaks: < 2
            - num_negative_peaks: < 1
            - peak_to_trough_duration: 0.1-1.15 ms
            - waveform_baseline_flatness: < 0.5
            - peak_after_to_trough_ratio: < 0.8
            - exp_decay: 0.01-0.1

        mua (spike quality - any failure -> "mua"):
            - amplitude_median: > 30 uV (absolute value)
            - snr: > 5
            - amplitude_cutoff: < 0.2
            - num_spikes: > 300
            - rpv: < 0.1 (refractory period violations)
            - presence_ratio: > 0.7
            - drift_ptp: < 100 um
            - isolation_distance: > 20 (if computed)
            - l_ratio: < 0.3 (if computed)

        non-somatic (waveform shape):
            - peak_before_to_trough_ratio: < 3
            - peak_before_width: > 0.15 ms
            - trough_width: > 0.2 ms
            - peak_before_to_peak_after_ratio: < 3
            - main_peak_to_trough_ratio: < 0.8

        To modify thresholds:
            thresholds = bombcell_get_default_thresholds()
            thresholds["mua"]["rpv"]["less"] = 0.05  # stricter RPV
            thresholds["mua"]["num_spikes"]["greater"] = 100  # lower spike count

        To disable a threshold:
            thresholds["mua"]["drift_ptp"] = {"greater": None, "less": None}

    Plotting Options
    ----------------
    plot_histograms : bool, default: True
        Plot metric histograms with threshold lines.
    plot_waveforms : bool, default: True
        Plot waveforms grouped by label.
    plot_upset : bool, default: True
        Plot UpSet plots showing metric failure combinations.
    waveform_ylims : tuple or None, default: (-300, 100)
        Y-axis limits for waveform plots.
    figsize_histograms : tuple, default: (15, 10)
        Figure size for histogram plot.

    Rerun Options
    -------------
    rerun_quality_metrics : bool, default: False
        Force recomputation of quality metrics.
    rerun_pca : bool, default: False
        Force recomputation of PCA (only if compute_distance_metrics=True).

    Job Parameters
    --------------
    n_jobs : int, default: -1
        Number of parallel jobs.
    progress_bar : bool, default: True
        Show progress bars.

    Returns
    -------
    labels : pd.DataFrame
        DataFrame with unit_ids as index and "bombcell_label" column.
        Labels: "good", "mua", "noise", "non_soma" (or "non_soma_good"/"non_soma_mua").
    metrics : pd.DataFrame
        Combined quality metrics and template metrics.
    figures : dict
        Dictionary of matplotlib figures: {"histograms", "waveforms", "upset"}.
    """
    import pandas as pd

    from .bombcell_curation import bombcell_get_default_thresholds, bombcell_label_units, save_bombcell_results

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Default QM params
    default_qm_params = {
        "presence_ratio": {"bin_duration_s": 60},
        "rp_violation": {"refractory_period_ms": 2.0, "censored_period_ms": 0.1},
        "sliding_rp_violation": {
            "exclude_ref_period_below_ms": 0.5,
            "max_ref_period_ms": 10.0,
            "confidence_threshold": 0.9,
        },
        "drift": {"interval_s": 60, "min_spikes_per_interval": 100},
    }
    if qm_params is not None:
        for key, val in qm_params.items():
            default_qm_params[key] = val
    qm_params = default_qm_params

    # Build metric names list
    metric_names = [
        "amplitude_median",
        "snr",
        "amplitude_cutoff",
        "num_spikes",
        "presence_ratio",
        "firing_rate",
    ]

    if rp_method == "sliding_rp":
        metric_names.append("sliding_rp_violation")
    else:
        metric_names.append("rp_violation")

    if compute_drift:
        metric_names.append("drift")

    if compute_distance_metrics:
        metric_names.append("mahalanobis")
        if not sorting_analyzer.has_extension("principal_components") or rerun_pca:
            if sorting_analyzer.has_extension("principal_components"):
                sorting_analyzer.delete_extension("principal_components")
            sorting_analyzer.compute(
                "principal_components", n_components=5, mode="by_channel_local", **job_kwargs
            )

    if use_valid_periods and not sorting_analyzer.has_extension("amplitude_scalings"):
        sorting_analyzer.compute("amplitude_scalings", **job_kwargs)

    # Compute quality metrics
    if sorting_analyzer.has_extension("quality_metrics") and rerun_quality_metrics:
        sorting_analyzer.delete_extension("quality_metrics")

    if not sorting_analyzer.has_extension("quality_metrics"):
        sorting_analyzer.compute(
            "quality_metrics", metric_names=metric_names, metric_params=qm_params, **job_kwargs
        )

    # Get thresholds
    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()

    # Run BombCell labeling
    labels = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
        thresholds=thresholds,
        label_non_somatic=label_non_somatic,
        split_non_somatic_good_mua=split_non_somatic_good_mua,
        use_valid_periods=use_valid_periods,
        valid_periods_params=valid_periods_params,
        **job_kwargs,
    )

    metrics = sorting_analyzer.get_metrics_extension_data()

    # Generate plots
    figures = {}

    if plot_histograms or plot_waveforms or plot_upset:
        import spikeinterface.widgets as sw

        if plot_histograms:
            w = sw.plot_metric_histograms(sorting_analyzer, thresholds, figsize=figsize_histograms)
            figures["histograms"] = w.figure

        if plot_waveforms:
            w = sw.plot_unit_labels(sorting_analyzer, labels["bombcell_label"], ylims=waveform_ylims)
            figures["waveforms"] = w.figure

        if plot_upset:
            w = sw.plot_bombcell_labels_upset(
                sorting_analyzer,
                unit_labels=labels["bombcell_label"],
                thresholds=thresholds,
                unit_labels_to_plot=["noise", "mua"],
            )
            figures["upset"] = w.figures

    # Save results
    if output_folder is not None:
        save_bombcell_results(
            metrics=metrics,
            unit_label=labels["bombcell_label"].values,
            thresholds=thresholds,
            folder=output_folder,
        )

        # Save figures
        if "histograms" in figures:
            figures["histograms"].savefig(output_folder / "metric_histograms.png", dpi=150, bbox_inches="tight")
        if "waveforms" in figures:
            figures["waveforms"].savefig(output_folder / "waveforms_by_label.png", dpi=150, bbox_inches="tight")
        if "upset" in figures:
            for i, fig in enumerate(figures["upset"]):
                fig.savefig(output_folder / f"upset_plot_{i}.png", dpi=150, bbox_inches="tight")

    print(f"Labeled {len(labels)} units:")
    print(labels["bombcell_label"].value_counts().to_string())

    return labels, metrics, figures
