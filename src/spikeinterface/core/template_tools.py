from __future__ import annotations
import numpy as np
import warnings

from .template import Templates
from .waveform_extractor import WaveformExtractor
from .sparsity import compute_sparsity, _sparsity_doc
from .recording_tools import get_channel_distances, get_noise_levels


def _get_dense_templates_array(templates_or_waveform_extractor):
    if isinstance(templates_or_waveform_extractor, Templates):
        templates_array = templates_or_waveform_extractor.get_dense_templates()
    elif isinstance(templates_or_waveform_extractor, WaveformExtractor):
        templates_array = templates_or_waveform_extractor.get_all_templates(mode="average")
    else:
        raise ValueError("templates_or_waveform_extractor should be Templates or WaveformExtractor")
    return templates_array


def get_template_amplitudes(
    templates_or_waveform_extractor,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" = "extremum",
):
    """
    Get amplitude per channel for each unit.

    Parameters
    ----------
    templates_or_waveform_extractor: Templates | WaveformExtractor
        A Templates or a WaveformExtractor object
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels
    mode: "extremum" | "at_index", default: "extremum"
        "extremum":  max or min
        "at_index": take value at spike index

    Returns
    -------
    peak_values: dict
        Dictionary with unit ids as keys and template amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos"), "'peak_sign' must be 'both', 'neg', or 'pos'"
    assert mode in ("extremum", "at_index"), "'mode' must be 'extremum' or 'at_index'"

    unit_ids = templates_or_waveform_extractor.unit_ids
    before = templates_or_waveform_extractor.nbefore

    peak_values = {}

    templates_array = _get_dense_templates_array(templates_or_waveform_extractor)

    for unit_ind, unit_id in enumerate(unit_ids):
        template = templates_array[unit_ind, :, :]

        if mode == "extremum":
            if peak_sign == "both":
                values = np.max(np.abs(template), axis=0)
            elif peak_sign == "neg":
                values = -np.min(template, axis=0)
            elif peak_sign == "pos":
                values = np.max(template, axis=0)
        elif mode == "at_index":
            if peak_sign == "both":
                values = np.abs(template[before, :])
            elif peak_sign == "neg":
                values = -template[before, :]
            elif peak_sign == "pos":
                values = template[before, :]

        peak_values[unit_id] = values

    return peak_values


def get_template_extremum_channel(
    templates_or_waveform_extractor,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" = "extremum",
    outputs: "id" | "index" = "id",
):
    """
    Compute the channel with the extremum peak for each unit.

    Parameters
    ----------
    templates_or_waveform_extractor: Templates | WaveformExtractor
        A Templates or a WaveformExtractor object
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels
    mode: "extremum" | "at_index", default: "extremum"
        "extremum":  max or min
        "at_index": take value at spike index
    outputs: "id" | "index", default: "id"
        * "id": channel id
        * "index": channel index

    Returns
    -------
    extremum_channels: dict
        Dictionary with unit ids as keys and extremum channels (id or index based on "outputs")
        as values
    """
    assert peak_sign in ("both", "neg", "pos")
    assert mode in ("extremum", "at_index")
    assert outputs in ("id", "index")

    unit_ids = templates_or_waveform_extractor.unit_ids
    channel_ids = templates_or_waveform_extractor.channel_ids

    peak_values = get_template_amplitudes(templates_or_waveform_extractor, peak_sign=peak_sign, mode=mode)
    extremum_channels_id = {}
    extremum_channels_index = {}
    for unit_id in unit_ids:
        max_ind = np.argmax(peak_values[unit_id])
        extremum_channels_id[unit_id] = channel_ids[max_ind]
        extremum_channels_index[unit_id] = max_ind

    if outputs == "id":
        return extremum_channels_id
    elif outputs == "index":
        return extremum_channels_index


def get_template_channel_sparsity(
    templates_or_waveform_extractor,
    method="radius",
    peak_sign="neg",
    num_channels=5,
    radius_um=100.0,
    threshold=5,
    by_property=None,
    outputs="id",
):
    """
    Get channel sparsity (subset of channels) for each template with several methods.

    Parameters
    ----------
    templates_or_waveform_extractor: Templates | WaveformExtractor
        A Templates or a WaveformExtractor object

    {}

    outputs: str
        * "id": channel id
        * "index": channel index

    Returns
    -------
    sparsity: dict
        Dictionary with unit ids as keys and sparse channel ids or indices (id or index based on "outputs")
        as values
    """
    from spikeinterface.core.sparsity import compute_sparsity

    warnings.warn(
        "The 'get_template_channel_sparsity()' function is deprecated. " "Use 'compute_sparsity()' instead",
        DeprecationWarning,
        stacklevel=2,
    )

    assert outputs in ("id", "index"), "'outputs' can either be 'id' or 'index'"
    sparsity = compute_sparsity(
        templates_or_waveform_extractor,
        method=method,
        peak_sign=peak_sign,
        num_channels=num_channels,
        radius_um=radius_um,
        threshold=threshold,
        by_property=by_property,
    )

    # handle output ids or indexes
    if outputs == "id":
        return sparsity.unit_id_to_channel_ids
    elif outputs == "index":
        return sparsity.unit_id_to_channel_indices


get_template_channel_sparsity.__doc__ = get_template_channel_sparsity.__doc__.format(_sparsity_doc)


def get_template_extremum_channel_peak_shift(
    templates_or_waveform_extractor, peak_sign: "neg" | "pos" | "both" = "neg"
):
    """
    In some situations spike sorters could return a spike index with a small shift related to the waveform peak.
    This function estimates and return these alignment shifts for the mean template.
    This function is internally used by `compute_spike_amplitudes()` to accurately retrieve the spike amplitudes.

    Parameters
    ----------
    templates_or_waveform_extractor: Templates | WaveformExtractor
        A Templates or a WaveformExtractor object
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels

    Returns
    -------
    shifts: dict
        Dictionary with unit ids as keys and shifts as values
    """
    unit_ids = templates_or_waveform_extractor.unit_ids
    channel_ids = templates_or_waveform_extractor.channel_ids
    nbefore = templates_or_waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(templates_or_waveform_extractor, peak_sign=peak_sign)

    shifts = {}

    templates_array = _get_dense_templates_array(templates_or_waveform_extractor)

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
    templates_or_waveform_extractor,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" = "at_index",
):
    """
    Computes amplitudes on the best channel.

    Parameters
    ----------
    templates_or_waveform_extractor: Templates | WaveformExtractor
        A Templates or a WaveformExtractor object
    peak_sign:  "neg" | "pos" | "both"
        Sign of the template to compute best channels
    mode: "extremum" | "at_index", default: "at_index"
        Where the amplitude is computed
        "extremum":  max or min
        "at_index": take value at spike index

    Returns
    -------
    amplitudes: dict
        Dictionary with unit ids as keys and amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos"), "'peak_sign' must be  'neg' or 'pos' or 'both'"
    assert mode in ("extremum", "at_index"), "'mode' must be 'extremum' or 'at_index'"
    unit_ids = templates_or_waveform_extractor.unit_ids
    channel_ids = templates_or_waveform_extractor.channel_ids

    before = templates_or_waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(
        templates_or_waveform_extractor, peak_sign=peak_sign, mode=mode
    )

    extremum_amplitudes = get_template_amplitudes(templates_or_waveform_extractor, peak_sign=peak_sign, mode=mode)

    unit_amplitudes = {}
    for unit_id in unit_ids:
        channel_id = extremum_channels_ids[unit_id]
        best_channel = list(channel_ids).index(channel_id)
        unit_amplitudes[unit_id] = extremum_amplitudes[unit_id][best_channel]

    return unit_amplitudes
