from __future__ import annotations
import numpy as np

from .template import Templates
from .sortinganalyzer import SortingAnalyzer


def get_dense_templates_array(one_object: Templates | SortingAnalyzer, return_scaled: bool = True):
    """
    Return dense templates as numpy array from either a Templates object or a SortingAnalyzer.

    Parameters
    ----------
    one_object : Templates | SortingAnalyzer
        The Templates or SortingAnalyzer objects. If SortingAnalyzer, it needs the "templates" extension.
    return_scaled : bool, default: True
        If True, templates are scaled.

    Returns
    -------
    dense_templates : np.ndarray
        The dense templates (num_units, num_samples, num_channels)
    """
    if isinstance(one_object, Templates):
        if return_scaled != one_object.is_scaled:
            raise ValueError(
                f"get_dense_templates_array: return_scaled={return_scaled} is not possible Templates has the reverse"
            )
        templates_array = one_object.get_dense_templates()
    elif isinstance(one_object, SortingAnalyzer):
        if return_scaled != one_object.return_scaled:
            raise ValueError(
                f"get_dense_templates_array: return_scaled={return_scaled} is not possible SortingAnalyzer has the reverse"
            )
        ext = one_object.get_extension("templates")
        if ext is not None:
            if "average" in ext.data:
                templates_array = ext.data.get("average")
            elif "median" in ext.data:
                templates_array = ext.data.get("median")
            else:
                raise ValueError("Average or median templates have not been computed.")
        else:
            raise ValueError("SortingAnalyzer need extension 'templates' to be computed to retrieve templates")
    else:
        raise ValueError("Input should be Templates or SortingAnalyzer")

    return templates_array


def _get_nbefore(one_object):
    if isinstance(one_object, Templates):
        return one_object.nbefore
    elif isinstance(one_object, SortingAnalyzer):
        ext = one_object.get_extension("templates")
        if ext is None:
            raise ValueError("SortingAnalyzer need extension 'templates' to be computed")
        return ext.nbefore
    else:
        raise ValueError("Input should be Templates or SortingAnalyzer")


def get_template_amplitudes(
    templates_or_sorting_analyzer,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" | "peak_to_peak" = "extremum",
    return_scaled: bool = True,
    abs_value: bool = True,
):
    """
    Get amplitude per channel for each unit.

    Parameters
    ----------
    templates_or_sorting_analyzer : Templates | SortingAnalyzer
        A Templates or a SortingAnalyzer object
    peak_sign :  "neg" | "pos" | "both"
        Sign of the template to find extremum channels
    mode : "extremum" | "at_index" | "peak_to_peak", default: "at_index"
        Where the amplitude is computed
        * "extremum" : take the peak value (max or min depending on `peak_sign`)
        * "at_index" : take value at `nbefore` index
        * "peak_to_peak" : take the peak-to-peak amplitude
    return_scaled : bool, default True
        The amplitude is scaled or not.
    abs_value : bool = True
        Whether the extremum amplitude should be returned as an absolute value or not

    Returns
    -------
    peak_values : dict
        Dictionary with unit ids as keys and template amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos"), "'peak_sign' must be 'both', 'neg', or 'pos'"
    assert mode in ("extremum", "at_index", "peak_to_peak"), "'mode' must be 'extremum', 'at_index', or 'peak_to_peak'"

    unit_ids = templates_or_sorting_analyzer.unit_ids
    before = _get_nbefore(templates_or_sorting_analyzer)

    peak_values = {}

    templates_array = get_dense_templates_array(templates_or_sorting_analyzer, return_scaled=return_scaled)

    for unit_ind, unit_id in enumerate(unit_ids):
        template = templates_array[unit_ind, :, :]

        if mode == "extremum":
            if peak_sign == "both":
                values = np.max(np.abs(template), axis=0)
            elif peak_sign == "neg":
                values = np.min(template, axis=0)
            elif peak_sign == "pos":
                values = np.max(template, axis=0)
        elif mode == "at_index":
            if peak_sign == "both":
                values = np.abs(template[before, :])
            elif peak_sign in ["neg", "pos"]:
                values = template[before, :]
        elif mode == "peak_to_peak":
            values = np.ptp(template, axis=0)

        if abs_value:
            values = np.abs(values)

        peak_values[unit_id] = values

    return peak_values


def get_template_extremum_channel(
    templates_or_sorting_analyzer,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" | "peak_to_peak" = "extremum",
    outputs: "id" | "index" = "id",
):
    """
    Compute the channel with the extremum peak for each unit.

    Parameters
    ----------
    templates_or_sorting_analyzer : Templates | SortingAnalyzer
        A Templates or a SortingAnalyzer object
    peak_sign :  "neg" | "pos" | "both"
        Sign of the template to find extremum channels
    mode : "extremum" | "at_index" | "peak_to_peak", default: "at_index"
        Where the amplitude is computed
        * "extremum" : take the peak value (max or min depending on `peak_sign`)
        * "at_index" : take value at `nbefore` index
        * "peak_to_peak" : take the peak-to-peak amplitude
    outputs : "id" | "index", default: "id"
        * "id" : channel id
        * "index" : channel index

    Returns
    -------
    extremum_channels : dict
        Dictionary with unit ids as keys and extremum channels (id or index based on "outputs")
        as values
    """
    assert peak_sign in ("both", "neg", "pos"), "`peak_sign` must be one of `both`, `neg`, or `pos`"
    assert mode in ("extremum", "at_index", "peak_to_peak"), "'mode' must be 'extremum', 'at_index', or 'peak_to_peak'"
    assert outputs in ("id", "index"), "`outputs` must be either `id` or `index`"

    unit_ids = templates_or_sorting_analyzer.unit_ids
    channel_ids = templates_or_sorting_analyzer.channel_ids

    # if SortingAnalyzer need to use global SortingAnalyzer return_scaled otherwise
    # we use the Templates is_scaled
    if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
        return_scaled = templates_or_sorting_analyzer.return_scaled
    else:
        return_scaled = templates_or_sorting_analyzer.is_scaled

    peak_values = get_template_amplitudes(
        templates_or_sorting_analyzer, peak_sign=peak_sign, mode=mode, return_scaled=return_scaled
    )
    extremum_channels_id = {}
    extremum_channels_index = {}
    for unit_id in unit_ids:
        max_ind = np.argmax(np.abs(peak_values[unit_id]))
        extremum_channels_id[unit_id] = channel_ids[max_ind]
        extremum_channels_index[unit_id] = max_ind

    if outputs == "id":
        return extremum_channels_id
    elif outputs == "index":
        return extremum_channels_index


def get_template_extremum_channel_peak_shift(templates_or_sorting_analyzer, peak_sign: "neg" | "pos" | "both" = "neg"):
    """
    In some situations spike sorters could return a spike index with a small shift related to the waveform peak.
    This function estimates and return these alignment shifts for the mean template.
    This function is internally used by `compute_spike_amplitudes()` to accurately retrieve the spike amplitudes.

    Parameters
    ----------
    templates_or_sorting_analyzer : Templates | SortingAnalyzer
        A Templates or a SortingAnalyzer object
    peak_sign :  "neg" | "pos" | "both"
        Sign of the template to find extremum channels

    Returns
    -------
    shifts : dict
        Dictionary with unit ids as keys and shifts as values
    """
    unit_ids = templates_or_sorting_analyzer.unit_ids
    channel_ids = templates_or_sorting_analyzer.channel_ids
    nbefore = _get_nbefore(templates_or_sorting_analyzer)

    extremum_channels_ids = get_template_extremum_channel(templates_or_sorting_analyzer, peak_sign=peak_sign)

    shifts = {}

    # We need to use the SortingAnalyzer return_scaled
    # We need to use the Templates is_scaled
    if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
        return_scaled = templates_or_sorting_analyzer.return_scaled
    else:
        return_scaled = templates_or_sorting_analyzer.is_scaled

    templates_array = get_dense_templates_array(templates_or_sorting_analyzer, return_scaled=return_scaled)

    for unit_ind, unit_id in enumerate(unit_ids):
        template = templates_array[unit_ind, :, :]

        chan_id = extremum_channels_ids[unit_id]
        chan_ind = list(channel_ids).index(chan_id)

        if peak_sign == "both":
            peak_pos = np.argmax(np.abs(template[:, chan_ind]))
        elif peak_sign == "neg":
            peak_pos = np.argmin(template[:, chan_ind])
        elif peak_sign == "pos":
            peak_pos = np.argmax(template[:, chan_ind])
        shift = peak_pos - nbefore
        shifts[unit_id] = shift

    return shifts


def get_template_extremum_amplitude(
    templates_or_sorting_analyzer,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" | "peak_to_peak" = "at_index",
    abs_value: bool = True,
):
    """
    Computes amplitudes on the best channel.

    Parameters
    ----------
    templates_or_sorting_analyzer : Templates | SortingAnalyzer
        A Templates or a SortingAnalyzer object
    peak_sign :  "neg" | "pos" | "both"
        Sign of the template to find extremum channels
    mode : "extremum" | "at_index" | "peak_to_peak", default: "at_index"
        Where the amplitude is computed
        * "extremum": take the peak value (max or min depending on `peak_sign`)
        * "at_index": take value at `nbefore` index
        * "peak_to_peak": take the peak-to-peak amplitude
    abs_value : bool = True
        Whether the extremum amplitude should be returned as an absolute value or not


    Returns
    -------
    amplitudes : dict
        Dictionary with unit ids as keys and amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos"), "'peak_sign' must be  'neg' or 'pos' or 'both'"
    assert mode in ("extremum", "at_index", "peak_to_peak"), "'mode' must be 'extremum', 'at_index', or 'peak_to_peak'"
    unit_ids = templates_or_sorting_analyzer.unit_ids
    channel_ids = templates_or_sorting_analyzer.channel_ids

    extremum_channels_ids = get_template_extremum_channel(templates_or_sorting_analyzer, peak_sign=peak_sign, mode=mode)

    if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
        return_scaled = templates_or_sorting_analyzer.return_scaled
    else:
        return_scaled = templates_or_sorting_analyzer.is_scaled

    extremum_amplitudes = get_template_amplitudes(
        templates_or_sorting_analyzer, peak_sign=peak_sign, mode=mode, return_scaled=return_scaled, abs_value=abs_value
    )

    unit_amplitudes = {}
    for unit_id in unit_ids:
        channel_id = extremum_channels_ids[unit_id]
        best_channel = list(channel_ids).index(channel_id)
        unit_amplitudes[unit_id] = extremum_amplitudes[unit_id][best_channel]

    return unit_amplitudes
