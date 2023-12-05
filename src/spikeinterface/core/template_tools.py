from __future__ import annotations
import numpy as np
import warnings

from .sparsity import compute_sparsity, _sparsity_doc
from .recording_tools import get_channel_distances, get_noise_levels
from .template import Templates


def get_template_amplitudes(
    waveform_extractor, peak_sign: "neg" | "pos" | "both" = "neg", mode: "extremum" | "at_index" = "extremum"
):
    """
    Get amplitude per channel for each unit.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
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
    unit_ids = waveform_extractor.sorting.unit_ids

    before = waveform_extractor.nbefore

    peak_values = {}

    templates = waveform_extractor.get_all_templates(mode="average")
    for unit_ind, unit_id in enumerate(unit_ids):
        template = templates[unit_ind, :, :]

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
    waveform_extractor,
    peak_sign: "neg" | "pos" | "both" = "neg",
    mode: "extremum" | "at_index" = "extremum",
    outputs: "id" | "index" = "id",
):
    """
    Compute the channel with the extremum peak for each unit.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
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

    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.channel_ids

    peak_values = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign, mode=mode)
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
    waveform_extractor,
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
        waveform_extractor: WaveformExtractor
            The waveform extractor
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
        waveform_extractor,
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


def get_template_extremum_channel_peak_shift(waveform_extractor, peak_sign: "neg" | "pos" | "both" = "neg"):
    """
    In some situations spike sorters could return a spike index with a small shift related to the waveform peak.
    This function estimates and return these alignment shifts for the mean template.
    This function is internally used by `compute_spike_amplitudes()` to accurately retrieve the spike amplitudes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels

    Returns
    -------
    shifts: dict
        Dictionary with unit ids as keys and shifts as values
    """
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign)

    shifts = {}

    templates = waveform_extractor.get_all_templates(mode="average")
    for unit_ind, unit_id in enumerate(unit_ids):
        template = templates[unit_ind, :, :]

        chan_id = extremum_channels_ids[unit_id]
        chan_ind = waveform_extractor.channel_ids_to_indices([chan_id])[0]

        if peak_sign == "both":
            peak_pos = np.argmax(np.abs(template[:, chan_ind]))
        elif peak_sign == "neg":
            peak_pos = np.argmin(template[:, chan_ind])
        elif peak_sign == "pos":
            peak_pos = np.argmax(template[:, chan_ind])
        shift = peak_pos - waveform_extractor.nbefore
        shifts[unit_id] = shift

    return shifts


def get_template_extremum_amplitude(
    waveform_extractor, peak_sign: "neg" | "pos" | "both" = "neg", mode: "extremum" | "at_index" = "at_index"
):
    """
    Computes amplitudes on the best channel.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
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
    unit_ids = waveform_extractor.sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(waveform_extractor, peak_sign=peak_sign, mode=mode)

    extremum_amplitudes = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign, mode=mode)

    unit_amplitudes = {}
    for unit_id in unit_ids:
        channel_id = extremum_channels_ids[unit_id]
        best_channel = waveform_extractor.channel_ids_to_indices([channel_id])[0]
        unit_amplitudes[unit_id] = extremum_amplitudes[unit_id][best_channel]

    return unit_amplitudes


def move_templates(templates, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates given some displacements using spatial interpolation (cubic or linear).
    Optionally can be remapped to another probe with a diffrents geometry.

    Before interpolation this function make is checking that templates can be interpolated on new
    positions using the sparsity information : the entire sparse template must be covered at all new positions.
    If the Templates object is dense then no check is possible and a wrning is emited.

    Parameters
    ----------
    templates: Templates
        A numpy array with dense templates.
        shape = (num_template, num_sample, num_channel)
    displacements: np.array
        Displacement vector
        shape: (num_displacement, 2)
    source_probe: Probe
        The Probe object on which templates are defined
    dest_probe: Probe or None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method: str, default "cubic"
        The interpolation method.
    interpolation_method: str, default "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates: list of Templates

    """
    if templates.are_templates_sparse():
        # TODO make some check!!
        pass
    else:
        warnings.warn("move_templates() with dense templates cannot checks that template can be interpolated at all places")
    
    dense_templates = templates.get_dense_templates()
    dense_templates_moved = move_dense_templates(templates, displacements, source_probe, dest_probe=None, interpolation_method="cubic")

    new_templates = []
    for i in range(dense_templates_moved.shape[0]):
        dense_templates = Templates(
            templates_array=dense_templates_moved,
            sampling_frequency=templates.sampling_frequency,
            nbefore=templates.nbefore,
            channel_ids=templates.channel_ids,
            unit_ids=templates.unit_ids,
        )
        # TODO : sparsify back the templates
        new_templates.append(dense_templates)

    return new_templates







def move_dense_templates(templates, displacements, source_probe, dest_probe=None, interpolation_method="cubic"):
    """
    Move all templates given some displacements using spatial interpolation (cubic or linear).
    Optionally can be remapped to another probe with a diffrents geometry.

    This function operate on dense template.

    Note : in this function no check are done if template can be interpolable after displacements.
    To make some check use teh higher level function move_templates().

    Parameters
    ----------
    templates: np.array
        A numpy array with dense templates.
        shape = (num_template, num_sample, num_channel)
    displacements: np.array
        Displacement vector
        shape: (num_displacement, 2)
    source_probe: Probe
        The Probe object on which templates are defined
    dest_probe: Probe or None
        The destination Probe. Can be different geometry than the original.
        If None then the same probe  is used.
    interpolation_method: str, default "cubic"
        The interpolation method.
    interpolation_method: str, default "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates: np.array
        shape = (num_displacement, num_template, num_sample, num_channel, )
    """
    assert displacements.ndim == 2
    assert displacements.shape[1] == 2

    if dest_probe is None:
        dest_probe = source_probe
    src_channel_locations = source_probe.contact_positions
    dest_channel_locations = dest_probe.contact_positions
    moved_locations = dest_channel_locations[np.newaxis, :, :] - displacements.reshape(-1, 1, 2)
    templates_moved = interpolate_templates(templates, src_channel_locations, moved_locations, interpolation_method=interpolation_method)
    return templates_moved





def interpolate_templates(templates, source_locations, dest_locations, interpolation_method="cubic"):
    """
    Interpolate template to a new position.
    Uselfull to create motion or to remap template form probeA to probeB.

    Note that several moves can be done by broadcating when dest_locations have one more dim.

    Parameters
    ----------
    templates: np.array
        A numpy array with dense templates.
        shape = (num_template, num_sample, num_channel)
    source_locations: np.array 
        The channel source location corresponding to templates.
        shape = (num_channel, 2)
    dest_locations: np.array
        The new channel position, if ndim == 3, then the interpolation is broadcated with last dim.
        shape = (num_channel, 2) or (num_motion, num_channel, 2)
    interpolation_method: str, default "cubic"
        The interpolation method.
    
    Returns
    -------
    new_templates: np.array
        shape = (num_template, num_sample, num_channel) or = (num_motion, num_template, num_sample, num_channel, )
    """
    import scipy.interpolate

    source_locations = np.asarray(source_locations)
    dest_locations = np.asarray(dest_locations)
    if dest_locations.ndim == 2:
        new_shape = templates.shape
    elif dest_locations.ndim == 3:
        new_shape = (dest_locations.shape[0], ) + templates.shape
    else:
        raise ValueError("Bad dest_locations")

    new_templates = np.zeros(new_shape, dtype=templates.dtype)
    
    for template_ind in range(templates.shape[0]):
        for time_ind in range(templates.shape[1]):
            template = templates[template_ind, time_ind, :]
            interp_template = scipy.interpolate.griddata(source_locations, template, dest_locations,
                                                         method=interpolation_method,
                                                         fill_value=0)
            if dest_locations.ndim == 2:
                new_templates[template_ind, time_ind, :] = interp_template
            elif dest_locations.ndim == 3: 
                new_templates[:, template_ind, time_ind, :] = interp_template

    return new_templates