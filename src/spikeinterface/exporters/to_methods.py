"""
Export a methods section for academic papers from a SortingAnalyzer.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import spikeinterface


# Citations for SpikeInterface, sorters, and analysis tools
CITATIONS = {
    "spikeinterface": (
        "Buccino, A. P., Hurwitz, C. L., Garcia, S., Magland, J., Siegle, J. H., Hurwitz, R., & Hennig, M. H. "
        "(2020). SpikeInterface, a unified framework for spike sorting. eLife, 9, e61834. "
        "https://doi.org/10.7554/eLife.61834"
    ),
    "bombcell": (
        "Fabre, J. M. J., van Beest, E. H., Peters, A. J., Carandini, M., & Harris, K. D. (2023). "
        "Bombcell: automated curation and cell classification of spike-sorted electrophysiology data. "
        "Zenodo. https://doi.org/10.5281/zenodo.8172821"
    ),
    "kilosort": (
        "Pachitariu, M., Steinmetz, N. A., Kadir, S. N., Carandini, M., & Harris, K. D. (2016). "
        "Fast and accurate spike sorting of high-channel count probes with KiloSort. "
        "Advances in Neural Information Processing Systems, 29, 4448-4456."
    ),
    "kilosort2": (
        "Pachitariu, M., Steinmetz, N. A., Kadir, S. N., Carandini, M., & Harris, K. D. (2016). "
        "Fast and accurate spike sorting of high-channel count probes with KiloSort. "
        "Advances in Neural Information Processing Systems, 29, 4448-4456."
    ),
    "kilosort2_5": (
        "Pachitariu, M., Steinmetz, N. A., Kadir, S. N., Carandini, M., & Harris, K. D. (2016). "
        "Fast and accurate spike sorting of high-channel count probes with KiloSort. "
        "Advances in Neural Information Processing Systems, 29, 4448-4456."
    ),
    "kilosort3": (
        "Pachitariu, M., Steinmetz, N. A., Kadir, S. N., Carandini, M., & Harris, K. D. (2016). "
        "Fast and accurate spike sorting of high-channel count probes with KiloSort. "
        "Advances in Neural Information Processing Systems, 29, 4448-4456."
    ),
    "kilosort4": (
        "Pachitariu, M., Sridhar, S., Pennington, J., & Stringer, C. (2024). "
        "Spike sorting with Kilosort4. Nature Methods. https://doi.org/10.1038/s41592-024-02232-7"
    ),
    "mountainsort4": (
        "Chung, J. E., Magland, J. F., Barnett, A. H., Tolosa, V. M., Tooker, A. C., Lee, K. Y., ... & Greengard, L. F. "
        "(2017). A fully automated approach to spike sorting. Neuron, 95(6), 1381-1394."
    ),
    "mountainsort5": (
        "Magland, J., Jun, J. J., Lovero, E., Morber, A. J., Barnett, A. H., Greengard, L. F., & Chung, J. E. (2020). "
        "SpikeForest, reproducible web-facing ground-truth validation of automated neural spike sorters. eLife, 9, e55167."
    ),
    "spykingcircus": (
        "Yger, P., Spampinato, G. L., Esposito, E., Lefebvre, B., Deny, S., Gardella, C., ... & Marre, O. (2018). "
        "A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. "
        "eLife, 7, e34518."
    ),
    "spykingcircus2": (
        "Yger, P., Spampinato, G. L., Esposito, E., Lefebvre, B., Deny, S., Gardella, C., ... & Marre, O. (2018). "
        "A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. "
        "eLife, 7, e34518."
    ),
    "tridesclous": (
        "Garcia, S., & Bhumbra, G. S. (2020). Tridesclous: a free, easy-to-use and lightweight spike sorter. "
        "FENS Forum 2020."
    ),
    "tridesclous2": (
        "Garcia, S., & Bhumbra, G. S. (2020). Tridesclous: a free, easy-to-use and lightweight spike sorter. "
        "FENS Forum 2020."
    ),
    "herdingspikes": (
        "Hilgen, G., Sorbaro, M., Pirber, S., Zber, J. E., Resber, M. E., Hennig, M. H., & Sernagor, E. (2017). "
        "Unsupervised spike sorting for large-scale, high-density multielectrode arrays. Cell Reports, 18(10), 2521-2532."
    ),
    "ironclust": (
        "Jun, J. J., Steinmetz, N. A., Siegle, J. H., Denman, D. J., Bauza, M., Barbarits, B., ... & Harris, T. D. (2017). "
        "Fully integrated silicon probes for high-density recording of neural activity. Nature, 551(7679), 232-236."
    ),
}

# Human-readable names for preprocessing classes
PREPROCESSING_NAMES = {
    "BandpassFilterRecording": "Bandpass Filter",
    "HighpassFilterRecording": "Highpass Filter",
    "LowpassFilterRecording": "Lowpass Filter",
    "NotchFilterRecording": "Notch Filter",
    "FilterRecording": "Filter",
    "CommonReferenceRecording": "Common Reference",
    "WhitenRecording": "Whitening",
    "NormalizeByQuantileRecording": "Normalize by Quantile",
    "ScaleRecording": "Scale",
    "CenterRecording": "Center",
    "ZScoreRecording": "Z-Score",
    "RectifyRecording": "Rectify",
    "ClipRecording": "Clip",
    "BlankSaturationRecording": "Blank Saturation",
    "RemoveArtifactsRecording": "Remove Artifacts",
    "RemoveBadChannelsRecording": "Remove Bad Channels",
    "InterpolateBadChannelsRecording": "Interpolate Bad Channels",
    "DepthOrderRecording": "Depth Order",
    "ResampleRecording": "Resample",
    "DecimateRecording": "Decimate",
    "PhaseShiftRecording": "Phase Shift",
    "AsTypeRecording": "Convert Data Type",
    "UnsignedToSignedRecording": "Unsigned to Signed",
    "AverageAcrossDirectionRecording": "Average Across Direction",
    "DirectionalDerivativeRecording": "Directional Derivative",
    "HighpassSpatialFilterRecording": "Highpass Spatial Filter",
    "GaussianBandpassFilterRecording": "Gaussian Bandpass Filter",
    "SilencedPeriodsRecording": "Silenced Periods",
    "CorrectMotionRecording": "Motion Correction",
    "InterpolateBadChannelsRecording": "Interpolate Bad Channels",
}

# Key parameters to show for each preprocessing step (for standard detail level)
PREPROCESSING_KEY_PARAMS = {
    "BandpassFilterRecording": ["freq_min", "freq_max", "filter_order"],
    "HighpassFilterRecording": ["freq_min", "filter_order"],
    "LowpassFilterRecording": ["freq_max", "filter_order"],
    "NotchFilterRecording": ["freq", "q"],
    "FilterRecording": ["band", "btype", "filter_order"],
    "CommonReferenceRecording": ["reference", "operator"],
    "WhitenRecording": ["mode", "radius_um"],
    "NormalizeByQuantileRecording": ["q1", "q2"],
    "ScaleRecording": ["gain", "offset"],
    "ResampleRecording": ["resample_rate"],
    "DecimateRecording": ["decimation_factor"],
    "PhaseShiftRecording": ["inter_sample_shift"],
    "RemoveBadChannelsRecording": ["bad_channel_ids"],
    "CorrectMotionRecording": ["spatial_interpolation_method"],
}


def _trace_preprocessing_chain(recording) -> list[dict]:
    """
    Walk the recording parent chain and extract preprocessing step info.

    Parameters
    ----------
    recording : BaseRecording
        The recording to trace

    Returns
    -------
    list[dict]
        List of dicts with 'class_name' and 'kwargs' for each step,
        ordered from original recording to most recent preprocessing
    """
    chain = []
    current = recording

    while current is not None:
        class_name = current.__class__.__name__
        kwargs = getattr(current, "_kwargs", {})

        # Filter out the 'recording' key as it's the parent reference
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "recording"}

        chain.append({"class_name": class_name, "kwargs": filtered_kwargs})
        current = current.get_parent()

    # Reverse so original recording is first
    chain.reverse()
    return chain


def _get_sorter_info(sorting) -> dict | None:
    """
    Extract sorter name, version, and parameters from a sorting.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object

    Returns
    -------
    dict | None
        Dict with sorter info, or None if not available
    """
    sorting_info = sorting.sorting_info
    if sorting_info is None:
        return None

    info = {}

    # Get sorter name and params
    params = sorting_info.get("params", {})
    info["sorter_name"] = params.get("sorter_name", "Unknown")
    info["sorter_params"] = params.get("sorter_params", {})

    # Get log info
    log = sorting_info.get("log", {})
    info["sorter_version"] = log.get("sorter_version", "Unknown")
    info["run_time"] = log.get("run_time")
    info["datetime"] = log.get("datetime")

    return info


def _format_value(value) -> str:
    """Format a parameter value for display."""
    if value is None:
        return "None"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, float):
        if value == float("inf"):
            return "infinity"
        elif value == float("-inf"):
            return "-infinity"
        else:
            # Format with reasonable precision
            return f"{value:g}"
    elif isinstance(value, (list, tuple)):
        if len(value) <= 5:
            return ", ".join(_format_value(v) for v in value)
        else:
            return f"[{len(value)} items]"
    elif isinstance(value, dict):
        return f"{{...}}"
    else:
        return str(value)


def _format_params_markdown(params: dict, detail_level: str, key_params: list | None = None) -> str:
    """Format parameters as markdown list."""
    lines = []

    if detail_level == "brief":
        return ""

    if detail_level == "standard" and key_params:
        # Only show key parameters
        for key in key_params:
            if key in params:
                lines.append(f"   - {key}: {_format_value(params[key])}")
    else:
        # Show all parameters (detailed)
        for key, value in params.items():
            lines.append(f"   - `{key}`: {_format_value(value)}")

    return "\n".join(lines)


def _format_params_text(params: dict, detail_level: str, key_params: list | None = None) -> str:
    """Format parameters as plain text."""
    lines = []

    if detail_level == "brief":
        return ""

    if detail_level == "standard" and key_params:
        for key in key_params:
            if key in params:
                lines.append(f"     {key}: {_format_value(params[key])}")
    else:
        for key, value in params.items():
            lines.append(f"     {key}: {_format_value(value)}")

    return "\n".join(lines)


def _format_params_latex(params: dict, detail_level: str, key_params: list | None = None) -> str:
    """Format parameters as LaTeX itemize."""
    lines = []

    if detail_level == "brief":
        return ""

    if detail_level == "standard" and key_params:
        params_to_show = {k: v for k, v in params.items() if k in key_params}
    else:
        params_to_show = params

    if params_to_show:
        lines.append("    \\begin{itemize}")
        for key, value in params_to_show.items():
            escaped_key = key.replace("_", "\\_")
            lines.append(f"      \\item \\texttt{{{escaped_key}}}: {_format_value(value)}")
        lines.append("    \\end{itemize}")

    return "\n".join(lines)


def _get_probe_description(sorting_analyzer) -> str:
    """Get a description of the probe."""
    try:
        probe = sorting_analyzer.get_probe()
        if probe is not None:
            manufacturer = probe.annotations.get("manufacturer", "")
            probe_name = probe.annotations.get("probe_name", "")
            if manufacturer and probe_name:
                return f"{manufacturer} {probe_name}"
            elif probe_name:
                return probe_name
            else:
                return "electrode array"
    except Exception:
        pass
    return "electrode array"


def _get_recording_duration(sorting_analyzer) -> float | None:
    """Get total recording duration in seconds."""
    try:
        total_samples = sum(sorting_analyzer.get_num_samples(i) for i in range(sorting_analyzer.get_num_segments()))
        return total_samples / sorting_analyzer.sampling_frequency
    except Exception:
        return None


def _describe_preprocessing_step(class_name: str, kwargs: dict, detail_level: str) -> str:
    """Generate a prose description of a preprocessing step."""
    human_name = PREPROCESSING_NAMES.get(class_name, class_name.replace("Recording", ""))

    # Build description based on the preprocessing type
    if "Filter" in class_name:
        freq_min = kwargs.get("freq_min") or kwargs.get("band", [None, None])[0] if isinstance(kwargs.get("band"), (list, tuple)) else None
        freq_max = kwargs.get("freq_max") or kwargs.get("band", [None, None])[1] if isinstance(kwargs.get("band"), (list, tuple)) else None
        order = kwargs.get("filter_order", kwargs.get("order"))
        ftype = kwargs.get("ftype", "butterworth")

        if freq_min and freq_max:
            desc = f"bandpass filtered ({freq_min}-{freq_max} Hz"
        elif freq_min:
            desc = f"highpass filtered (>{freq_min} Hz"
        elif freq_max:
            desc = f"lowpass filtered (<{freq_max} Hz"
        else:
            desc = f"filtered ("

        if detail_level == "detailed" and order:
            desc += f", {order}th order {ftype})"
        else:
            desc += ")"
        return desc

    elif "CommonReference" in class_name:
        ref = kwargs.get("reference", "global")
        operator = kwargs.get("operator", "median")
        if detail_level == "detailed":
            return f"re-referenced using {ref} {operator} referencing"
        return f"common {operator} referenced"

    elif "Whiten" in class_name:
        mode = kwargs.get("mode", "global")
        if detail_level == "detailed":
            radius = kwargs.get("radius_um")
            if radius:
                return f"whitened ({mode} mode, {radius} µm radius)"
        return "whitened"

    elif "Normalize" in class_name or "ZScore" in class_name:
        return "normalized"

    elif "RemoveBadChannels" in class_name or "InterpolateBadChannels" in class_name:
        return "with bad channels removed/interpolated"

    elif "Resample" in class_name:
        rate = kwargs.get("resample_rate")
        if rate:
            return f"resampled to {rate} Hz"
        return "resampled"

    elif "CorrectMotion" in class_name:
        method = kwargs.get("spatial_interpolation_method", "")
        if detail_level == "detailed" and method:
            return f"motion corrected (using {method} interpolation)"
        return "motion corrected"

    elif "PhaseShift" in class_name:
        return "phase shift corrected"

    elif "InjectTemplates" in class_name or "NoiseGenerator" in class_name:
        # These are used for synthetic/test data generation, not real preprocessing
        return None

    else:
        return human_name.lower()


def _describe_sorter_params(sorter_name: str, params: dict, detail_level: str) -> str:
    """Generate a prose description of key sorter parameters."""
    if not params or detail_level == "brief":
        return ""

    # Define key parameters for each sorter
    key_params_by_sorter = {
        "kilosort4": ["Th_universal", "Th_learned", "do_CAR", "batch_size", "nblocks"],
        "kilosort3": ["Th", "ThPre", "lam", "AUCsplit", "minFR"],
        "kilosort2": ["Th", "ThPre", "lam", "AUCsplit", "minFR"],
        "kilosort2_5": ["Th", "ThPre", "lam", "AUCsplit", "minFR"],
        "mountainsort5": ["scheme", "detect_threshold", "snippet_T1", "snippet_T2"],
        "spykingcircus2": ["detection", "selection", "clustering", "matching"],
        "tridesclous2": ["detection", "selection", "clustering"],
    }

    sorter_key = sorter_name.lower().replace("-", "").replace("_", "")
    key_params = key_params_by_sorter.get(sorter_key, [])

    if detail_level == "standard" and key_params:
        # Only describe key parameters in prose
        parts = []
        for key in key_params:
            if key in params:
                parts.append(f"{key}={_format_value(params[key])}")
        if parts:
            return " (" + ", ".join(parts) + ")"
        return ""
    elif detail_level == "detailed":
        # List all parameters
        parts = [f"{k}={_format_value(v)}" for k, v in params.items()]
        if parts:
            return ". Key parameters: " + ", ".join(parts)
        return ""
    return ""


def _describe_quality_metrics(params: dict, detail_level: str) -> str:
    """Generate a prose description of quality metrics computed."""
    metric_names = params.get("metric_names") or params.get("metrics_to_compute", [])
    if isinstance(metric_names, (list, tuple)):
        if detail_level == "brief":
            return "quality metrics"
        elif len(metric_names) <= 5 or detail_level == "detailed":
            return f"quality metrics ({', '.join(metric_names)})"
        else:
            return f"quality metrics ({len(metric_names)} metrics including {', '.join(metric_names[:3])}, etc.)"
    return "quality metrics"


def export_to_methods(
    sorting_analyzer,
    output_file: str | Path | None = None,
    format: str = "markdown",
    include_citations: bool = True,
    detail_level: str = "detailed",
    sorter_name: str | None = None,
    sorter_version: str | None = None,
    probe_name: str | None = None,
    probe_manufacturer: str | None = None,
    preprocessing_description: str | None = None,
) -> str:
    """
    Generate a methods section describing the spike sorting pipeline.

    This function extracts information from a SortingAnalyzer about the
    preprocessing steps, spike sorting parameters, and post-processing
    analyses that were performed, and formats them as a methods section
    suitable for academic papers.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object containing the sorting results and metadata
    output_file : str | Path | None, default: None
        If provided, write the methods section to this file
    format : str, default: "markdown"
        Output format: "markdown", "latex", or "text"
    include_citations : bool, default: True
        If True, include citation references at the end
    detail_level : str, default: "detailed"
        Level of detail: "brief" (just step names), "standard" (key parameters),
        or "detailed" (all parameters)
    sorter_name : str | None, default: None
        Override the sorter name if not available in sorting_info.
        Use this when loading sorted data from Phy/Kilosort output directly.
        Examples: "Kilosort4", "Kilosort2.5", "MountainSort5", "SpykingCircus2"
    sorter_version : str | None, default: None
        Override the sorter version if not available in sorting_info.
    probe_name : str | None, default: None
        Override the probe name if not set in probe annotations.
        Examples: "Neuropixels 1.0", "Neuropixels 2.0", "Cambridge NeuroTech H2"
    probe_manufacturer : str | None, default: None
        Override the probe manufacturer if not set in probe annotations.
        Examples: "IMEC", "Cambridge NeuroTech", "NeuroNexus"
    preprocessing_description : str | None, default: None
        Manual description of preprocessing if not done via SpikeInterface.
        Example: "bandpass filtered (300-6000 Hz) and common median referenced"

    Returns
    -------
    str
        The generated methods section text

    Notes
    -----
    For best results, ensure your data has complete metadata:

    - **Probe info**: Set via `recording.set_probe()` with annotations, or use
      `probe_name` and `probe_manufacturer` parameters
    - **Sorter info**: Automatically captured when using `spikeinterface.sorters.run_sorter()`.
      When loading from Phy/Kilosort output directly, use `sorter_name` parameter
    - **Preprocessing**: Automatically tracked when using `spikeinterface.preprocessing`.
      Otherwise, use `preprocessing_description` parameter

    Examples
    --------
    >>> # When sorter info is captured automatically (via run_sorter)
    >>> export_to_methods(sorting_analyzer)

    >>> # When loading from Kilosort output directly
    >>> export_to_methods(
    ...     sorting_analyzer,
    ...     sorter_name="Kilosort4",
    ...     sorter_version="4.0.1",
    ...     probe_name="Neuropixels 1.0",
    ...     probe_manufacturer="IMEC"
    ... )
    """
    if format not in ("markdown", "latex", "text"):
        raise ValueError(f"format must be 'markdown', 'latex', or 'text', got '{format}'")
    if detail_level not in ("brief", "standard", "detailed"):
        raise ValueError(f"detail_level must be 'brief', 'standard', or 'detailed', got '{detail_level}'")

    paragraphs = []
    citations_to_include = ["spikeinterface"]
    missing_info = []  # Track what info is missing

    si_version = spikeinterface.__version__

    # === Gather all information first ===
    fs = sorting_analyzer.sampling_frequency
    n_channels = sorting_analyzer.get_num_channels()
    duration = _get_recording_duration(sorting_analyzer)
    n_units = sorting_analyzer.get_num_units()

    # Get probe description - use override or extract from data
    if probe_name:
        if probe_manufacturer:
            probe_desc = f"{probe_manufacturer} {probe_name}"
        else:
            probe_desc = probe_name
    else:
        probe_desc = _get_probe_description(sorting_analyzer)
        if probe_desc == "electrode array":
            missing_info.append("probe_name")

    # Get preprocessing chain
    preprocessing_steps = []
    if sorting_analyzer.has_recording():
        recording = sorting_analyzer.recording
        chain = _trace_preprocessing_chain(recording)
        preprocessing_steps = [step for step in chain if step["class_name"].endswith("Recording") and step["kwargs"]]

    # Get sorter info - use overrides if provided
    sorter_info = _get_sorter_info(sorting_analyzer.sorting)

    # Apply overrides to sorter info
    if sorter_name:
        if sorter_info is None:
            sorter_info = {"sorter_name": sorter_name, "sorter_version": sorter_version or "", "sorter_params": {}, "run_time": None}
        else:
            sorter_info["sorter_name"] = sorter_name
            if sorter_version:
                sorter_info["sorter_version"] = sorter_version
    elif sorter_info is None:
        missing_info.append("sorter_name")

    # Get extensions
    if sorting_analyzer.format == "memory":
        extensions = sorting_analyzer.get_loaded_extension_names()
    else:
        extensions = sorting_analyzer.get_saved_extension_names()

    # Check for quality/template metrics for Bombcell citation
    has_quality_metrics = "quality_metrics" in extensions or "template_metrics" in extensions
    if has_quality_metrics:
        citations_to_include.append("bombcell")

    # === Build the methods section as prose ===

    # Title/Header
    if format == "markdown":
        paragraphs.append("## Spike Sorting Methods\n")
    elif format == "latex":
        paragraphs.append("\\section{Spike Sorting Methods}\n")
    else:
        paragraphs.append("SPIKE SORTING METHODS\n")

    # === First paragraph: Data acquisition and preprocessing ===
    para1_parts = []

    # Data acquisition sentence
    acq_sentence = f"Extracellular recordings were acquired at {fs:.0f} Hz using a {probe_desc} ({n_channels} channels"
    if duration is not None:
        if duration >= 60:
            acq_sentence += f", {duration/60:.1f} minutes of data"
        else:
            acq_sentence += f", {duration:.1f} seconds of data"
    acq_sentence += ")."
    para1_parts.append(acq_sentence)

    # Preprocessing sentence(s) - use manual description if provided
    if preprocessing_description:
        para1_parts.append(f"Raw voltage traces were {preprocessing_description}.")
    elif preprocessing_steps:
        prep_descriptions = []
        for step in preprocessing_steps:
            desc = _describe_preprocessing_step(step["class_name"], step["kwargs"], detail_level)
            if desc:
                prep_descriptions.append(desc)

        if prep_descriptions:
            if len(prep_descriptions) == 1:
                prep_sentence = f"Raw voltage traces were {prep_descriptions[0]}."
            elif len(prep_descriptions) == 2:
                prep_sentence = f"Raw voltage traces were {prep_descriptions[0]} and {prep_descriptions[1]}."
            else:
                prep_sentence = f"Raw voltage traces were {', '.join(prep_descriptions[:-1])}, and {prep_descriptions[-1]}."
            para1_parts.append(prep_sentence)
        else:
            missing_info.append("preprocessing")
    else:
        missing_info.append("preprocessing")

    paragraphs.append(" ".join(para1_parts))
    paragraphs.append("")

    # === Second paragraph: Spike sorting ===
    para2_parts = []

    if sorter_info:
        sorter_name = sorter_info["sorter_name"]
        sorter_version = sorter_info["sorter_version"]
        sorter_params = sorter_info["sorter_params"]

        # Add citation for this sorter
        sorter_key = sorter_name.lower().replace("-", "").replace("_", "")
        if sorter_key in CITATIONS:
            citations_to_include.append(sorter_key)

        # Build sorter description
        if format == "markdown":
            sort_sentence = f"Spike sorting was performed using **{sorter_name}**"
        elif format == "latex":
            sort_sentence = f"Spike sorting was performed using \\textbf{{{sorter_name}}}"
        else:
            sort_sentence = f"Spike sorting was performed using {sorter_name}"

        if sorter_version and sorter_version != "Unknown":
            sort_sentence += f" (version {sorter_version})"

        # Add parameter description
        param_desc = _describe_sorter_params(sorter_name, sorter_params, detail_level)
        sort_sentence += param_desc

        if not sort_sentence.endswith("."):
            sort_sentence += "."
        para2_parts.append(sort_sentence)

        # Add runtime info if available
        if detail_level == "detailed" and sorter_info.get("run_time") is not None:
            run_time = sorter_info["run_time"]
            if run_time >= 60:
                para2_parts.append(f"Sorting completed in {run_time/60:.1f} minutes.")
            else:
                para2_parts.append(f"Sorting completed in {run_time:.1f} seconds.")
    else:
        para2_parts.append("Spike sorting was performed (sorter parameters not recorded).")

    # Add unit count
    para2_parts.append(f"A total of {n_units} units were identified.")

    paragraphs.append(" ".join(para2_parts))
    paragraphs.append("")

    # === Third paragraph: Post-processing and quality control ===
    if extensions:
        para3_parts = []

        # Categorize extensions
        waveform_exts = [e for e in extensions if e in ("waveforms", "templates", "random_spikes")]
        location_exts = [e for e in extensions if "location" in e]
        metric_exts = [e for e in extensions if "metric" in e]
        other_exts = [e for e in extensions if e not in waveform_exts + location_exts + metric_exts]

        # Waveforms and templates
        if waveform_exts:
            wf_ext = sorting_analyzer.get_extension("waveforms")
            if wf_ext:
                ms_before = wf_ext.params.get("ms_before", 1)
                ms_after = wf_ext.params.get("ms_after", 2)
                para3_parts.append(f"Spike waveforms were extracted ({ms_before} ms before to {ms_after} ms after each spike) and averaged to compute unit templates.")

        # Quality metrics
        if "quality_metrics" in extensions:
            qm_ext = sorting_analyzer.get_extension("quality_metrics")
            if qm_ext:
                qm_desc = _describe_quality_metrics(qm_ext.params, detail_level)
                para3_parts.append(f"Unit {qm_desc} were computed to assess sorting quality.")

        if "template_metrics" in extensions:
            para3_parts.append("Template-based metrics were computed for each unit.")

        # Locations
        if "unit_locations" in extensions:
            loc_ext = sorting_analyzer.get_extension("unit_locations")
            method = loc_ext.params.get("method", "center_of_mass") if loc_ext else "center_of_mass"
            para3_parts.append(f"Unit locations were estimated using the {method.replace('_', ' ')} method.")

        # Other notable extensions
        if "principal_components" in extensions:
            pc_ext = sorting_analyzer.get_extension("principal_components")
            if pc_ext and detail_level == "detailed":
                n_comp = pc_ext.params.get("n_components", 5)
                para3_parts.append(f"Principal component analysis was performed ({n_comp} components).")

        if "correlograms" in extensions:
            para3_parts.append("Auto- and cross-correlograms were computed.")

        if "spike_amplitudes" in extensions:
            para3_parts.append("Spike amplitudes were extracted for each spike.")

        if para3_parts:
            paragraphs.append(" ".join(para3_parts))
            paragraphs.append("")

    # === Software attribution paragraph ===
    software_para = f"All spike sorting and analysis was performed using SpikeInterface version {si_version}"
    if has_quality_metrics:
        software_para += ", with quality metrics following the Bombcell framework"
    software_para += "."
    paragraphs.append(software_para)
    paragraphs.append("")

    # === Missing Info Warning ===
    if missing_info:
        paragraphs.append("")
        if format == "markdown":
            paragraphs.append("---")
            paragraphs.append("**Note**: Some information could not be extracted automatically and should be added manually:")
            for info in missing_info:
                if info == "probe_name":
                    paragraphs.append("- Probe type/name (use `probe_name` parameter)")
                elif info == "sorter_name":
                    paragraphs.append("- Spike sorter name and version (use `sorter_name` and `sorter_version` parameters)")
                elif info == "preprocessing":
                    paragraphs.append("- Preprocessing steps (use `preprocessing_description` parameter)")
            paragraphs.append("")
        elif format == "latex":
            paragraphs.append("\\textit{Note: Some information could not be extracted automatically. See function documentation for how to specify missing metadata.}")
            paragraphs.append("")
        else:
            paragraphs.append("NOTE: Missing information that should be added manually:")
            for info in missing_info:
                if info == "probe_name":
                    paragraphs.append("  - Probe type/name")
                elif info == "sorter_name":
                    paragraphs.append("  - Spike sorter name and version")
                elif info == "preprocessing":
                    paragraphs.append("  - Preprocessing steps")
            paragraphs.append("")

    # === Citations Section ===
    if include_citations:
        if format == "markdown":
            paragraphs.append("### References\n")
        elif format == "latex":
            paragraphs.append("\\subsection*{References}\n")
        else:
            paragraphs.append("References\n")

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for c in citations_to_include:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)

        for citation_key in unique_citations:
            if citation_key in CITATIONS:
                citation = CITATIONS[citation_key]
                if format == "markdown":
                    paragraphs.append(f"- {citation}\n")
                elif format == "latex":
                    paragraphs.append(f"\\bibitem{{{citation_key}}} {citation}\n")
                else:
                    paragraphs.append(f"- {citation}\n")

    # Join all paragraphs
    result = "\n".join(paragraphs)

    # Write to file if requested
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")

    return result
