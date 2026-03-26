"""
BombCell pipeline functions for preprocessing and quality control.

This module provides wrapper functions for running the full BombCell quality
control pipeline on spike-sorted data.

Functions
---------
get_default_preprocessing_params
    Get default parameters for preprocessing and SortingAnalyzer creation.
get_default_qc_params
    Get default parameters for quality metrics and BombCell labeling.
preprocess_for_bombcell
    Preprocess recording and create SortingAnalyzer with required extensions.
run_bombcell_qc
    Compute quality metrics, run BombCell labeling, and generate plots.

See Also
--------
bombcell_get_default_thresholds : Get default classification thresholds.
bombcell_label_units : Core labeling function.
"""

from __future__ import annotations
from pathlib import Path


def get_default_preprocessing_params():
    """
    Get default parameters for preprocessing and SortingAnalyzer creation.

    Returns a dictionary that can be modified and passed to preprocess_for_bombcell().

    Returns
    -------
    dict
        Dictionary with the following keys:

        **Highpass Filtering**

        freq_min : float, default: 300.0
            Highpass filter cutoff frequency in Hz. Removes low-frequency noise
            and LFP signals. Standard value for spike detection is 300 Hz.
            Lower values (150-250 Hz) may be used if spikes have significant
            low-frequency components.

        **Bad Channel Detection**

        detect_bad_channels : bool, default: True
            Whether to automatically detect and remove bad channels before
            further processing. Recommended to leave enabled.

        bad_channel_method : str, default: "coherence+psd"
            Method for detecting bad channels:
            - "coherence+psd": Combines coherence with neighbors and power
              spectral density analysis. Best for Neuropixels. (recommended)
            - "mad": Median absolute deviation of signal amplitude.
            - "std": Standard deviation based detection.

        **Phase Shift Correction**

        apply_phase_shift : bool, default: True
            Whether to apply inter-sample phase shift correction. Essential for
            Neuropixels probes where ADCs are multiplexed and channels are sampled
            at slightly different times. Corrects for timing offsets that can
            affect spike waveform shapes. Disable for non-Neuropixels probes.

        **Common Reference**

        apply_cmr : bool, default: True
            Whether to apply common reference to remove correlated noise.
            Highly recommended for Neuropixels recordings.

        cmr_reference : str, default: "global"
            Type of common reference:
            - "global": Use all channels (recommended for Neuropixels).
            - "local": Use nearby channels only (for probes with distinct groups).
            - "single": Reference to a single channel.

        cmr_operator : str, default: "median"
            Operation for computing reference signal:
            - "median": More robust to outliers (recommended).
            - "average": Standard mean reference.

        **SortingAnalyzer Settings**

        sparse : bool, default: True
            Use sparse waveform representation, storing only channels near each
            unit. Significantly reduces memory usage for high-channel-count probes.
            Recommended True for Neuropixels.

        return_in_uV : bool, default: True
            Convert waveforms to microvolts using gain/offset from probe metadata.
            Required for amplitude-based quality metrics to be meaningful.

        **Waveform Extraction**

        max_spikes_per_unit : int, default: 500
            Maximum number of spikes to extract per unit for waveform analysis.
            Higher values give better templates but use more memory/time.
            500 is typically sufficient for stable template estimation.

        ms_before : float, default: 3.0
            Milliseconds before spike peak to extract. 3.0 ms captures the
            pre-spike baseline and any pre-depolarization.

        ms_after : float, default: 3.0
            Milliseconds after spike peak to extract. 3.0 ms captures the
            repolarization and afterhyperpolarization.

        **Template Computation**

        template_operators : list, default: ["average", "median", "std"]
            Statistics to compute for templates:
            - "average": Mean waveform (standard template).
            - "median": Median waveform (robust to outliers).
            - "std": Standard deviation (waveform variability).

        include_multi_channel_metrics : bool, default: True
            Compute template metrics across multiple channels, including:
            - exp_decay: Exponential decay of amplitude across channels.
            - velocity: Propagation velocity estimate.
            Required for BombCell noise detection. Leave True.

    Examples
    --------
    >>> params = get_default_preprocessing_params()
    >>> params["freq_min"] = 250.0  # Lower cutoff for some cell types
    >>> params["max_spikes_per_unit"] = 1000  # More spikes for better templates
    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(recording, sorting, "analyzer.zarr", params=params)
    """
    return {
        # Highpass filtering
        "freq_min": 300.0,
        # Bad channel detection
        "detect_bad_channels": True,
        "bad_channel_method": "coherence+psd",
        # Phase shift (Neuropixels)
        "apply_phase_shift": True,
        # Common reference
        "apply_cmr": True,
        "cmr_reference": "global",
        "cmr_operator": "median",
        # SortingAnalyzer
        "sparse": True,
        "return_in_uV": True,
        # Waveforms
        "max_spikes_per_unit": 500,
        "ms_before": 3.0,
        "ms_after": 3.0,
        # Templates
        "template_operators": ["average", "median", "std"],
        "include_multi_channel_metrics": True,
    }


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

        rp_method : str, default: "sliding_rp"
            Method for computing refractory period violations:
            - "sliding_rp": IBL/Steinmetz method that sweeps across RP values
              and estimates contamination. More robust. (recommended)
            - "llobet": Single RP value method from Llobet et al.

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

        **Sliding RP Method Parameters** (used if rp_method="sliding_rp")

        sliding_rp_exclude_below_ms : float, default: 0.5
            Exclude refractory periods below this value when sweeping.
            Avoids artifacts from very short intervals.

        sliding_rp_max_ms : float, default: 10.0
            Maximum refractory period to test when sweeping.

        sliding_rp_confidence : float, default: 0.9
            Confidence level for contamination estimate. Higher values
            give more conservative (higher) contamination estimates.

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
    >>> # More lenient refractory period for fast-spiking neurons
    >>> params["refractory_period_ms"] = 1.5
    >>> labels, metrics, figs = run_bombcell_qc(analyzer, params=params)
    """
    return {
        # Which metrics to compute
        "compute_amplitude_cutoff": False,  # slow - requires spike_amplitudes
        "compute_distance_metrics": False,
        "compute_drift": True,
        "rp_method": "sliding_rp",
        # BombCell labeling options
        "label_non_somatic": True,
        "split_non_somatic_good_mua": False,
        "use_valid_periods": False,
        # Presence ratio
        "presence_ratio_bin_duration_s": 60,
        # Refractory period violations
        "refractory_period_ms": 2.0,
        "censored_period_ms": 0.1,
        # Sliding RP method
        "sliding_rp_exclude_below_ms": 0.5,
        "sliding_rp_max_ms": 10.0,
        "sliding_rp_confidence": 0.9,
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


def preprocess_for_bombcell(
    recording,
    sorting,
    analyzer_folder: str | Path,
    params: dict | None = None,
    rerun_extensions: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Preprocess recording and create SortingAnalyzer with extensions for BombCell.

    This function applies standard preprocessing steps (filtering, bad channel
    removal, phase shift correction, common reference) and creates a SortingAnalyzer
    with all extensions required for BombCell quality control.

    Parameters
    ----------
    recording : BaseRecording
        Raw recording to preprocess. Typically loaded with si.read_spikeglx()
        or si.read_openephys().
    sorting : BaseSorting
        Spike sorting result. Can be loaded with si.read_sorter_folder() or
        any other sorting loader.
    analyzer_folder : str or Path
        Path to save the SortingAnalyzer. Will be created in zarr format.
        If folder exists, loads existing analyzer instead of creating new one.
    params : dict or None, default: None
        Preprocessing parameters from get_default_preprocessing_params().
        If None, uses all default values.
    rerun_extensions : bool, default: False
        If True, recompute all extensions even if they already exist.
        Useful after changing parameters.
    n_jobs : int, default: -1
        Number of parallel jobs for computation. -1 uses all available CPUs.
    progress_bar : bool, default: True
        Show progress bars during computation.

    Returns
    -------
    analyzer : SortingAnalyzer
        SortingAnalyzer saved to analyzer_folder with computed extensions:
        random_spikes, waveforms, templates, noise_levels, unit_locations,
        spike_locations, template_metrics. Note: spike_amplitudes is computed
        on-demand by run_bombcell_qc() if compute_amplitude_cutoff=True.
    rec_preprocessed : BaseRecording
        Preprocessed recording (lazy chain, not saved to disk).
        Can be used for further analysis or passed to other functions.
    bad_channel_ids : list or None
        List of channel IDs that were detected as bad and removed.
        None if detect_bad_channels=False.

    Examples
    --------
    Basic usage with defaults:

    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(recording, sorting, "analyzer.zarr")

    With custom parameters:

    >>> params = get_default_preprocessing_params()
    >>> params["freq_min"] = 250.0
    >>> params["detect_bad_channels"] = False  # Already cleaned
    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(
    ...     recording, sorting, "analyzer.zarr", params=params
    ... )

    Rerun extensions after parameter change:

    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(
    ...     recording, sorting, "analyzer.zarr", rerun_extensions=True
    ... )
    """
    import spikeinterface.full as si

    if params is None:
        params = get_default_preprocessing_params()

    analyzer_folder = Path(analyzer_folder)
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    # Preprocess
    rec = si.highpass_filter(recording, freq_min=params["freq_min"])

    bad_channel_ids = None
    if params["detect_bad_channels"]:
        bad_channel_ids, _ = si.detect_bad_channels(rec, method=params["bad_channel_method"])
        bad_channel_ids = list(bad_channel_ids)
        if len(bad_channel_ids) > 0:
            rec = rec.remove_channels(bad_channel_ids)

    if params["apply_phase_shift"]:
        rec = si.phase_shift(rec)

    if params["apply_cmr"]:
        rec = si.common_reference(rec, reference=params["cmr_reference"], operator=params["cmr_operator"])

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
            sparse=params["sparse"],
            format="zarr",
            folder=analyzer_folder,
            return_in_uV=params["return_in_uV"],
        )

    # Compute extensions
    def _compute(name, **kwargs):
        if analyzer.has_extension(name) and not rerun_extensions:
            return
        if analyzer.has_extension(name):
            analyzer.delete_extension(name)
        analyzer.compute(name, **kwargs)

    _compute("random_spikes", method="uniform", max_spikes_per_unit=params["max_spikes_per_unit"])
    _compute("waveforms", ms_before=params["ms_before"], ms_after=params["ms_after"], **job_kwargs)
    _compute("templates", operators=list(params["template_operators"]))
    _compute("noise_levels")
    # spike_amplitudes computed on-demand by run_bombcell_qc if compute_amplitude_cutoff=True
    _compute("unit_locations")
    _compute("spike_locations", **job_kwargs)
    _compute("template_metrics", include_multi_channel_metrics=params["include_multi_channel_metrics"])

    return analyzer, rec_preprocessed, bad_channel_ids


def run_bombcell_qc(
    sorting_analyzer,
    output_folder: str | Path = "bombcell",
    params: dict | None = None,
    thresholds: dict | None = None,
    valid_periods_params: dict | None = None,
    rerun_quality_metrics: bool = False,
    rerun_pca: bool = False,
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
        Analyzer with template_metrics extension computed (from preprocess_for_bombcell).
    output_folder : str or Path, default: "bombcell"
        Folder to save results (CSV files and plots). Set to None to skip saving.
        Created if it doesn't exist.
    params : dict or None, default: None
        QC parameters from get_default_qc_params(). If None, uses defaults.
    thresholds : dict or None, default: None
        BombCell classification thresholds from bombcell_get_default_thresholds().
        If None, uses defaults. Structure:

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
    - metric_histograms.png: Histogram of each metric with threshold lines.
    - waveforms_by_label.png: Waveform overlays for each label category.
    - upset_plot_*.png: UpSet plots showing metric failure combinations.

    Examples
    --------
    Basic usage with defaults:

    >>> labels, metrics, figs = run_bombcell_qc(analyzer)

    With custom parameters and thresholds:

    >>> params = get_default_qc_params()
    >>> params["compute_distance_metrics"] = True  # For chronic recordings
    >>> params["refractory_period_ms"] = 1.5  # For fast-spiking neurons
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
    from .bombcell_curation import bombcell_get_default_thresholds, bombcell_label_units, save_bombcell_results

    if params is None:
        params = get_default_qc_params()

    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Build QM params
    qm_params = {
        "presence_ratio": {"bin_duration_s": params["presence_ratio_bin_duration_s"]},
        "rp_violation": {
            "refractory_period_ms": params["refractory_period_ms"],
            "censored_period_ms": params["censored_period_ms"],
        },
        "sliding_rp_violation": {
            "exclude_ref_period_below_ms": params["sliding_rp_exclude_below_ms"],
            "max_ref_period_ms": params["sliding_rp_max_ms"],
            "confidence_threshold": params["sliding_rp_confidence"],
        },
        "drift": {
            "interval_s": params["drift_interval_s"],
            "min_spikes_per_interval": params["drift_min_spikes"],
        },
    }

    # Build metric names
    metric_names = ["amplitude_median", "snr", "num_spikes", "presence_ratio", "firing_rate"]

    if params["compute_amplitude_cutoff"]:
        metric_names.append("amplitude_cutoff")
        # amplitude_cutoff requires spike_amplitudes or amplitude_scalings
        if not sorting_analyzer.has_extension("spike_amplitudes") and not sorting_analyzer.has_extension(
            "amplitude_scalings"
        ):
            sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

    if params["rp_method"] == "sliding_rp":
        metric_names.append("sliding_rp_violation")
    else:
        metric_names.append("rp_violation")

    if params["compute_drift"]:
        metric_names.append("drift")

    if params["compute_distance_metrics"]:
        metric_names.append("mahalanobis")
        if not sorting_analyzer.has_extension("principal_components") or rerun_pca:
            if sorting_analyzer.has_extension("principal_components"):
                sorting_analyzer.delete_extension("principal_components")
            sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)

    if params["use_valid_periods"] and not sorting_analyzer.has_extension("amplitude_scalings"):
        sorting_analyzer.compute("amplitude_scalings", **job_kwargs)

    # Compute quality metrics
    if sorting_analyzer.has_extension("quality_metrics") and rerun_quality_metrics:
        sorting_analyzer.delete_extension("quality_metrics")

    if not sorting_analyzer.has_extension("quality_metrics"):
        sorting_analyzer.compute("quality_metrics", metric_names=metric_names, metric_params=qm_params, **job_kwargs)

    # Run BombCell
    labels = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
        thresholds=thresholds,
        label_non_somatic=params["label_non_somatic"],
        split_non_somatic_good_mua=params["split_non_somatic_good_mua"],
        use_valid_periods=params["use_valid_periods"],
        valid_periods_params=valid_periods_params,
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
        save_bombcell_results(
            metrics=metrics,
            unit_label=labels["bombcell_label"].values,
            thresholds=thresholds,
            folder=output_folder,
        )
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
