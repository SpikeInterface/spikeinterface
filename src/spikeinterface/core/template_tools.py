from __future__ import annotations
import numpy as np

from .template import Templates
from .waveform_tools import estimate_templates_with_accumulator
from .sorting_tools import random_spikes_selection
from .sortinganalyzer import SortingAnalyzer

import warnings



def get_dense_templates_array(one_object: Templates | SortingAnalyzer, return_in_uV: bool = True):
    """
    Return dense templates as numpy array from either a Templates object or a SortingAnalyzer.

    Parameters
    ----------
    one_object : Templates | SortingAnalyzer
        The Templates or SortingAnalyzer objects. If SortingAnalyzer, it needs the "templates" extension.
    return_in_uV : bool, default: True
        If True, templates are scaled.

    Returns
    -------
    dense_templates : np.ndarray
        The dense templates (num_units, num_samples, num_channels)
    """
    if isinstance(one_object, Templates):
        if return_in_uV != one_object.is_in_uV:
            raise ValueError(
                f"get_dense_templates_array: return_in_uV={return_in_uV} is not possible Templates has the reverse"
            )
        templates_array = one_object.get_dense_templates()
    elif isinstance(one_object, SortingAnalyzer):
        if return_in_uV != one_object.return_in_uV:
            raise ValueError(
                f"get_dense_templates_array: return_in_uV={return_in_uV} is not possible SortingAnalyzer has the reverse"
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
    return_in_uV: bool = True,
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
    return_in_uV : bool, default True
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

    templates_array = get_dense_templates_array(templates_or_sorting_analyzer, return_in_uV=return_in_uV)

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



def _get_main_channel_from_template_array(templates_array, peak_mode, main_channel_peak_sign, nbefore):
    # Step1 : max on time axis
    if peak_mode == "extremum":
        if main_channel_peak_sign == "both":
            values = np.max(np.abs(templates_array), axis=1)
        elif main_channel_peak_sign == "neg":
            values = -np.min(templates_array, axis=1)
        elif main_channel_peak_sign == "pos":
            values = np.max(templates_array, axis=1)
    elif peak_mode == "at_index":
        if main_channel_peak_sign == "both":
            values = np.abs(templates_array[:, nbefore, :])
        elif main_channel_peak_sign in ["neg", "pos"]:
            values = templates_array[:, nbefore, :]
    elif peak_mode == "peak_to_peak":
        values = np.ptp(templates_array, axis=1)
    
    # Step2: max on channel axis
    main_channel_index = np.argmax(values, axis=1)

    return main_channel_index

def estimate_main_channel_from_recording(
        recording,
        sorting,
        main_channel_peak_sign: "neg" | "both" | "pos" = "both",
        peak_mode: "extremum" | "at_index" | "peak_to_peak" = "extremum",
        num_spikes_for_main_channel=100,
        ms_before = 1.0,
        ms_after = 2.5,
        seed=None,
        **job_kwargs
):
    """
    Estimate the main channel from recording using `estimate_templates_with_accumulator()`

    """

    if main_channel_peak_sign == "pos":
        warnings.warn(
            "estimate_main_channel_from_recording() with peak_sign='pos' is a strange case maybe you " \
            "should revert the traces instead"
        )


    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]
    random_spikes_indices = random_spikes_selection(
        sorting,
        num_samples,
        method="uniform",
        max_spikes_per_unit=num_spikes_for_main_channel,
        margin_size=max(nbefore, nafter),
        seed=seed,
    )
    spikes = sorting.to_spike_vector()
    spikes = spikes[random_spikes_indices]

    templates_array = estimate_templates_with_accumulator(
        recording,
        spikes,
        sorting.unit_ids,
        nbefore,
        nafter,
        return_in_uV=False,
        job_name="estimate_main_channel",
        **job_kwargs,
    )

    main_channel_index = _get_main_channel_from_template_array(templates_array, peak_mode, main_channel_peak_sign, nbefore)

    return main_channel_index





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
    warnings.warn("get_template_extremum_channel() is deprecated use analyzer.get_main_channels() instead")
    # TODO make a better logic here

    assert peak_sign in ("both", "neg", "pos"), "`peak_sign` must be one of `both`, `neg`, or `pos`"
    assert mode in ("extremum", "at_index", "peak_to_peak"), "'mode' must be 'extremum', 'at_index', or 'peak_to_peak'"
    assert outputs in ("id", "index"), "`outputs` must be either `id` or `index`"

    unit_ids = templates_or_sorting_analyzer.unit_ids
    channel_ids = templates_or_sorting_analyzer.channel_ids

    # if SortingAnalyzer need to use global SortingAnalyzer return_scaled otherwise
    # we use the Templates is_in_uV
    if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
        # For backward compatibility
        if hasattr(templates_or_sorting_analyzer, "return_scaled"):
            return_in_uV = templates_or_sorting_analyzer.return_scaled
        else:
            return_in_uV = templates_or_sorting_analyzer.return_in_uV
    else:
        return_in_uV = templates_or_sorting_analyzer.is_in_uV

    peak_values = get_template_amplitudes(
        templates_or_sorting_analyzer, peak_sign=peak_sign, mode=mode, return_in_uV=return_in_uV
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
    # We need to use the Templates is_in_uV
    if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
        # For backward compatibility
        if hasattr(templates_or_sorting_analyzer, "return_scaled"):
            return_in_uV = templates_or_sorting_analyzer.return_scaled
        else:
            return_in_uV = templates_or_sorting_analyzer.return_in_uV
    else:
        return_in_uV = templates_or_sorting_analyzer.is_in_uV

    templates_array = get_dense_templates_array(templates_or_sorting_analyzer, return_in_uV=return_in_uV)

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
        # For backward compatibility
        if hasattr(templates_or_sorting_analyzer, "return_scaled"):
            return_in_uV = templates_or_sorting_analyzer.return_scaled
        else:
            return_in_uV = templates_or_sorting_analyzer.return_in_uV
    else:
        return_in_uV = templates_or_sorting_analyzer.is_in_uV

    extremum_amplitudes = get_template_amplitudes(
        templates_or_sorting_analyzer, peak_sign=peak_sign, mode=mode, return_in_uV=return_in_uV, abs_value=abs_value
    )

    unit_amplitudes = {}
    for unit_id in unit_ids:
        channel_id = extremum_channels_ids[unit_id]
        best_channel = list(channel_ids).index(channel_id)
        unit_amplitudes[unit_id] = extremum_amplitudes[unit_id][best_channel]

    return unit_amplitudes
