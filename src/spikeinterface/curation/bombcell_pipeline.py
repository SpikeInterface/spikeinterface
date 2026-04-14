"""
BombCell pipeline functions for quality control.

This module provides wrapper functions for running the BombCell quality
control pipeline on spike-sorted data.

Functions
---------
get_default_qc_params
    Get default parameters for quality metrics and BombCell labeling.
run_bombcell_qc
    Compute quality metrics, run BombCell labeling, and generate plots.

See Also
--------
bombcell_get_default_thresholds : Get default classification thresholds.
bombcell_label_units : Core labeling function.
"""

from __future__ import annotations
from pathlib import Path


def get_default_qc_params():
    """
    Get default parameters for quality metrics and BombCell labeling.

    Returns a dictionary that can be modified and passed to run_bombcell_qc().

    Returns
    -------
    dict
        Dictionary with the following keys:

        **Quality Metrics Selection**

        compute_amplitude_cutoff : bool, default: False
            Whether to compute amplitude_cutoff metric (estimated percentage of
            missing spikes). Requires spike_amplitudes extension which is slow
            to compute for large recordings. When enabled, spike_amplitudes will
            be computed automatically if not already present.

        compute_distance_metrics : bool, default: False
            Whether to compute isolation_distance and l_ratio metrics.
            These require PCA computation and are slow for large datasets.
            Useful for chronic recordings where cluster stability matters.
            Not recommended for acute recordings with expected drift.

        compute_drift : bool, default: True
            Whether to compute drift metrics (drift_ptp, drift_std, drift_mad).
            Measures how much units move over the recording. Important for
            acute recordings. drift_ptp (peak-to-peak drift in um) is used
            by BombCell MUA thresholds.

        rp_method : dict, default: {"method": "sliding_rp", "exclude_ref_period_below_ms": 0.5, "max_ref_period_ms": 10.0}
            Refractory period violation method and its parameters.
            The "method" key selects the algorithm:
            - "sliding_rp": IBL/Steinmetz method that sweeps across RP values
              and estimates contamination. More robust. (recommended)
            - "llobet": Single RP value method from Llobet et al.
            Additional keys are passed as metric params for the selected method.

        **BombCell Labeling Options**

        label_non_somatic : bool, default: True
            Whether to detect and label non-somatic (axonal/dendritic) units.
            These have distinctive waveform shapes: narrow initial peak,
            often triphasic. Set False to skip this classification.

        split_non_somatic_good_mua : bool, default: False
            If True, split non-somatic units into "non_soma_good" and
            "non_soma_mua" based on whether they pass MUA thresholds.
            If False, all non-somatic units are labeled "non_soma".

        use_valid_periods : bool, default: False
            If True, identify valid time periods per unit (where the unit
            has stable amplitude and low refractory violations) and compute
            quality metrics only on those periods. Useful for recordings
            with unstable periods. Requires amplitude_scalings extension.

        **Presence Ratio Parameters**

        presence_ratio_bin_duration_s : float, default: 60
            Bin duration in seconds for computing presence ratio.
            Presence ratio = fraction of bins containing at least one spike.
            60s bins are standard; shorter bins are stricter.

        **Refractory Period Violation Parameters**

        refractory_period_ms : float, default: 2.0
            Refractory period duration in milliseconds. Spikes closer than
            this are considered violations. 2.0 ms is conservative; some
            fast-spiking neurons may need 1.0-1.5 ms.

        censored_period_ms : float, default: 0.1
            Censored period in milliseconds. Spikes within this period of
            each other are not counted (accounts for detection artifacts).
            0.1 ms is standard.

        **Drift Parameters**

        drift_interval_s : float, default: 60
            Interval in seconds for computing drift. Unit positions are
            estimated in each interval and drift is the movement across intervals.

        drift_min_spikes : int, default: 100
            Minimum spikes required per interval to estimate position.
            Intervals with fewer spikes are skipped.

        **Plotting Options**

        plot_histograms : bool, default: True
            Generate histograms of all metrics with threshold lines.
            Saved as "metric_histograms.png".

        plot_waveforms : bool, default: True
            Generate waveform overlay plots grouped by label (good, mua, noise, etc.).
            Saved as "waveforms_by_label.png".

        plot_upset : bool, default: True
            Generate UpSet plots showing which metrics fail together.
            Useful for understanding why units are labeled noise/mua.
            Requires 'upsetplot' package. Saved as "upset_plot_*.png".

        waveform_ylims : tuple or None, default: (-300, 100)
            Y-axis limits for waveform plots in microvolts.
            None for automatic scaling.

        figsize_histograms : tuple, default: (15, 10)
            Figure size (width, height) in inches for histogram plot.

    Examples
    --------
    >>> params = get_default_qc_params()
    >>> # Stricter for chronic recordings
    >>> params["compute_distance_metrics"] = True
    >>> params["compute_drift"] = False  # Less relevant for chronic
    >>> labels, metrics, figs = run_bombcell_qc(analyzer, params=params)
    """
    return {
        # Which metrics to compute
        "compute_amplitude_cutoff": False,  # slow - requires spike_amplitudes
        "compute_distance_metrics": False,
        "compute_drift": True,
        "rp_method": {
            "method": "sliding_rp",
            "exclude_ref_period_below_ms": 0.5,
            "max_ref_period_ms": 10.0,
        },
        # BombCell labeling options
        "label_non_somatic": True,
        "split_non_somatic_good_mua": False,
        "use_valid_periods": False,
        # Presence ratio
        "presence_ratio_bin_duration_s": 60,
        # Refractory period violations
        "refractory_period_ms": 2.0,
        "censored_period_ms": 0.1,
        # Drift
        "drift_interval_s": 60,
        "drift_min_spikes": 100,
        # Plotting
        "plot_histograms": True,
        "plot_waveforms": True,
        "plot_upset": True,
        "waveform_ylims": (-300, 100),
        "figsize_histograms": (15, 10),
    }


def run_bombcell_qc(
    sorting_analyzer,
    output_folder: str | Path = "bombcell",
    params: dict | str | Path | None = None,
    thresholds: dict | str | Path | None = None,
    valid_periods_params: dict | None = None,
    rerun_quality_metrics: bool = False,
    rerun_pca: bool = False,
    rerun_amplitude_scalings: bool = False,
    rerun_valid_periods: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Compute quality metrics and run BombCell unit labeling.

    This function computes quality metrics on the SortingAnalyzer, runs the
    BombCell labeling algorithm to classify units as good/mua/noise/non_soma,
    generates diagnostic plots, and saves results.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        Analyzer with template_metrics extension computed.
    output_folder : str or Path, default: "bombcell"
        Folder to save results (CSV files and plots). Set to None to skip saving.
        Created if it doesn't exist.
    params : dict, str, Path, or None, default: None
        QC parameters from get_default_qc_params(), or a path to a JSON file
        containing such a dict. If None, uses defaults.

        To override the default metric list built from the compute_* flags, set
        params["metric_names"] to an explicit list of metric names. To override
        the default metric params, set params["metric_params"] to a dict mapping
        metric name -> param dict. Any metric you add this way must correspond to
        a valid SpikeInterface quality metric.
    thresholds : dict, str, Path, or None, default: None
        BombCell classification thresholds from bombcell_get_default_thresholds(),
        or a path to a JSON file containing such a dict. If None, uses defaults.
        Structure:

        - "noise": Thresholds for waveform quality. Failing ANY -> "noise".
        - "mua": Thresholds for spike quality. Failing ANY -> "mua".
        - "non-somatic": Thresholds for waveform shape. Determines non-somatic units.

        Each threshold is {"greater": value, "less": value}. Use None to disable.
        See bombcell_get_default_thresholds() docstring for all thresholds.

    valid_periods_params : dict or None, default: None
        Parameters for valid_unit_periods extension if params["use_valid_periods"]=True.
        Keys: refractory_period_ms, censored_period_ms, period_mode,
        period_duration_s_absolute, minimum_valid_period_duration.
    rerun_quality_metrics : bool, default: False
        Force recomputation of quality metrics even if they exist.
    rerun_pca : bool, default: False
        Force recomputation of PCA (only relevant if compute_distance_metrics=True).
    rerun_amplitude_scalings : bool, default: False
        Force recomputation of amplitude_scalings (used as a prerequisite for
        amplitude_cutoff and valid periods).
    rerun_valid_periods : bool, default: False
        Force recomputation of valid_unit_periods (only relevant if use_valid_periods=True).
    n_jobs : int, default: -1
        Number of parallel jobs.
    progress_bar : bool, default: True
        Show progress bars.

    Returns
    -------
    labels : pd.DataFrame
        DataFrame with unit_ids as index and "bombcell_label" column.
        Possible labels: "good", "mua", "noise", "non_soma"
        (or "non_soma_good"/"non_soma_mua" if split_non_somatic_good_mua=True).
    metrics : pd.DataFrame
        Combined DataFrame of all quality metrics and template metrics.
        Index is unit_ids, columns are metric names.
    figures : dict
        Dictionary of matplotlib figures:
        - "histograms": Metric histograms with threshold lines.
        - "waveforms": Waveform overlays grouped by label.
        - "upset": List of UpSet plot figures (one per label type).

    Saved Files (in output_folder)
    ------------------------------
    - labeling_results_wide.csv: One row per unit with all metrics and label.
    - labeling_results_narrow.csv: One row per unit-metric with pass/fail status.
    - thresholds.json: Thresholds used for classification (reproducibility).
    - bombcell_config.json: bombcell-specific options (rp_method, label_non_somatic,
      split_non_somatic_good_mua, use_valid_periods, valid_periods_params).
      Note: quality metric params are stored on the analyzer via the quality_metrics extension.
    - metric_histograms.png: Histogram of each metric with threshold lines.
    - waveforms_by_label.png: Waveform overlays for each label category.
    - upset_plot_*.png: UpSet plots showing metric failure combinations.

    Note on valid periods
    ---------------------
    When ``use_valid_periods=True``, the valid time periods per unit are stored by
    the ``valid_unit_periods`` extension on the analyzer itself (npy on disk).
    Access them via ``sorting_analyzer.get_extension("valid_unit_periods").get_data()``.

    Examples
    --------
    Basic usage with defaults:

    >>> labels, metrics, figs = run_bombcell_qc(analyzer)

    With custom parameters and thresholds:

    >>> params = get_default_qc_params()
    >>> params["compute_distance_metrics"] = True  # For chronic recordings
    >>>
    >>> thresholds = bombcell_get_default_thresholds()
    >>> thresholds["mua"]["rpv"]["less"] = 0.05  # Stricter RP violations
    >>> thresholds["mua"]["num_spikes"]["greater"] = 100  # Lower spike threshold
    >>>
    >>> labels, metrics, figs = run_bombcell_qc(
    ...     analyzer,
    ...     output_folder="qc_results",
    ...     params=params,
    ...     thresholds=thresholds,
    ... )

    Get good units for downstream analysis:

    >>> good_units = labels[labels["bombcell_label"] == "good"].index.tolist()
    >>> mua_units = labels[labels["bombcell_label"] == "mua"].index.tolist()
    """
    import json

    from .bombcell_curation import (
        bombcell_get_default_thresholds,
        bombcell_label_units,
        save_bombcell_results,
    )

    # Resolve params (dict, JSON path, or None)
    if params is None:
        params = get_default_qc_params()
    elif isinstance(params, (str, Path)):
        with open(params, "r") as f:
            params = json.load(f)

    # Resolve thresholds (dict, JSON path, or None)
    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()
    elif isinstance(thresholds, (str, Path)):
        with open(thresholds, "r") as f:
            thresholds = json.load(f)

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Build QM params
    rp_method_dict = params["rp_method"]
    rp_method_name = rp_method_dict["method"]
    rp_method_params = {k: v for k, v in rp_method_dict.items() if k != "method"}

    qm_params = {
        "presence_ratio": {"bin_duration_s": params["presence_ratio_bin_duration_s"]},
        "rp_violation": {
            "refractory_period_ms": params["refractory_period_ms"],
            "censored_period_ms": params["censored_period_ms"],
        },
        "drift": {
            "interval_s": params["drift_interval_s"],
            "min_spikes_per_interval": params["drift_min_spikes"],
        },
    }

    if rp_method_name == "sliding_rp":
        qm_params["sliding_rp_violation"] = rp_method_params
        rp_metric_name = "sliding_rp_violation"
    else:
        qm_params["rp_violation"].update(rp_method_params)
        rp_metric_name = "rp_violation"

    # Build metric names (user can override via params["metric_names"])
    if "metric_names" in params and params["metric_names"] is not None:
        metric_names = list(params["metric_names"])
    else:
        metric_names = ["amplitude_median", "snr", "num_spikes", "presence_ratio", "firing_rate"]

        if params["compute_amplitude_cutoff"]:
            metric_names.append("amplitude_cutoff")

        metric_names.append(rp_metric_name)

        if params["compute_drift"]:
            metric_names.append("drift")

        if params["compute_distance_metrics"]:
            metric_names.append("mahalanobis")

    # Ensure prerequisite extensions are computed for whichever metrics are requested
    # amplitude_cutoff uses amplitude_scalings (not spike_amplitudes) in this pipeline
    needs_amplitude_scalings = "amplitude_cutoff" in metric_names or params["use_valid_periods"]
    if needs_amplitude_scalings:
        if rerun_amplitude_scalings or not sorting_analyzer.has_extension("amplitude_scalings"):
            sorting_analyzer.compute("amplitude_scalings", **job_kwargs)

    if "mahalanobis" in metric_names:
        if not sorting_analyzer.has_extension("principal_components") or rerun_pca:
            sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)

    # User-provided metric_params override the defaults built above
    if "metric_params" in params and params["metric_params"] is not None:
        for metric, mp in params["metric_params"].items():
            qm_params[metric] = mp

    # Compute quality metrics
    if not sorting_analyzer.has_extension("quality_metrics") or rerun_quality_metrics:
        sorting_analyzer.compute("quality_metrics", metric_names=metric_names, metric_params=qm_params, **job_kwargs)

    # Run BombCell
    labels = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
        thresholds=thresholds,
        label_non_somatic=params["label_non_somatic"],
        split_non_somatic_good_mua=params["split_non_somatic_good_mua"],
        use_valid_periods=params["use_valid_periods"],
        valid_periods_params=valid_periods_params,
        quality_metric_params={"metric_names": metric_names, "metric_params": qm_params},
        rerun_valid_periods=rerun_valid_periods,
        **job_kwargs,
    )

    metrics = sorting_analyzer.get_metrics_extension_data()

    # Plots
    figures = {}
    if params["plot_histograms"] or params["plot_waveforms"] or params["plot_upset"]:
        import spikeinterface.widgets as sw

        if params["plot_histograms"]:
            w = sw.plot_metric_histograms(sorting_analyzer, thresholds, figsize=params["figsize_histograms"])
            figures["histograms"] = w.figure

        if params["plot_waveforms"]:
            w = sw.plot_unit_labels(sorting_analyzer, labels["bombcell_label"], ylims=params["waveform_ylims"])
            figures["waveforms"] = w.figure

        if params["plot_upset"]:
            w = sw.plot_bombcell_labels_upset(
                sorting_analyzer,
                unit_labels=labels["bombcell_label"],
                thresholds=thresholds,
                unit_labels_to_plot=["noise", "mua"],
            )
            figures["upset"] = w.figures

    # Save
    if output_folder is not None:
        import json

        save_bombcell_results(
            metrics=metrics,
            unit_label=labels["bombcell_label"].values,
            thresholds=thresholds,
            folder=output_folder,
        )
        # Save thresholds and bombcell-specific config so the run is reproducible
        # (quality metric params are stored on the analyzer itself via the extension)
        with open(output_folder / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
        bombcell_config = {
            "label_non_somatic": params["label_non_somatic"],
            "split_non_somatic_good_mua": params["split_non_somatic_good_mua"],
            "use_valid_periods": params["use_valid_periods"],
            "rp_method": params["rp_method"],
            "valid_periods_params": valid_periods_params,
        }
        with open(output_folder / "bombcell_config.json", "w") as f:
            json.dump(bombcell_config, f, indent=2)
        # Note: valid periods are stored by the valid_unit_periods extension on the analyzer itself.
        # Access them via: analyzer.get_extension("valid_unit_periods").get_data()
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
