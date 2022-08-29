import numpy as np
import numpy.lib.recfunctions as rfn

from ..core import get_channel_distances, get_noise_levels


def get_template_amplitudes(waveform_extractor, peak_sign: str = "neg", mode: str = "extremum"):
    """
    Get amplitude per channel for each unit.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    mode: str
        'extremum':  max or min
        'at_index': take value at spike index

    Returns
    -------
    peak_values: dict
        Dictionary with unit ids as keys and template amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos")
    assert mode in ("extremum", "at_index")
    unit_ids = waveform_extractor.sorting.unit_ids

    before = waveform_extractor.nbefore

    peak_values = {}

    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id, mode="average")

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


def get_template_extremum_channel(waveform_extractor, peak_sign: str = "neg", mode: str = "extremum", outputs: str = "id"):
    """
    Compute the channel with the extremum peak for each unit.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    mode: str
        'extremum':  max or min
        'at_index': take value at spike index
    outputs: str
        * 'id': channel id
        * 'index': channel index

    Returns
    -------
    extremum_channels: dict
        Dictionary with unit ids as keys and extremum channels (id or index based on 'outputs')
        as values
    """
    assert peak_sign in ("both", "neg", "pos")
    assert mode in ("extremum", "at_index")
    assert outputs in ("id", "index")

    unit_ids = waveform_extractor.sorting.unit_ids
    channel_ids = waveform_extractor.recording.channel_ids

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
    method="best_channels",
    peak_sign="neg",
    outputs="id",
    num_channels=None,
    radius_um=None,
    threshold=5,
    by_property=None,
):
    """
    Get channel sparsity (subset of channels) for each template with several methods.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    method: str
        * "best_channels": N best channels with the largest amplitude. Use the 'num_channels' argument to specify the
                         number of channels.
        * "radius": radius around the best channel. Use the 'radius_um' argument to specify the radius in um
        * "threshold": thresholds based on template signal-to-noise ratio. Use the 'threshold' argument
                       to specify the SNR threshold.
        * "by_property": sparsity is given by a property of the recording and sorting(e.g. 'group').
                         Use the 'by_property' argument to specify the property name.
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    outputs: str
        * 'id': channel id
        * 'index': channel index
    num_channels: int
        Number of channels for 'best_channels' method
    radius_um: float
        Radius in um for 'radius' method
    threshold: float
        Threshold in SNR 'threshold' method
    by_property: object
        Property name for 'by_property' method

    Returns
    -------
    sparsity: dict
        Dictionary with unit ids as keys and sparse channel ids or indices (id or index based on 'outputs')
        as values
    """
    assert method in ("best_channels", "radius", "threshold", "by_property")
    assert outputs in ("id", "index")
    we = waveform_extractor

    unit_ids = we.sorting.unit_ids
    channel_ids = we.recording.channel_ids

    sparsity_with_index = {}
    if method == "best_channels":
        assert num_channels is not None
        # take
        peak_values = get_template_amplitudes(we, peak_sign=peak_sign)
        for unit_id in unit_ids:
            chan_inds = np.argsort(np.abs(peak_values[unit_id]))[::-1]
            chan_inds = chan_inds[:num_channels]
            sparsity_with_index[unit_id] = chan_inds

    elif method == "radius":
        assert radius_um is not None
        best_chan = get_template_extremum_channel(we, outputs="index")
        distances = get_channel_distances(we.recording)
        for unit_id in unit_ids:
            chan_ind = best_chan[unit_id]
            (chan_inds,) = np.nonzero(distances[chan_ind, :] <= radius_um)
            sparsity_with_index[unit_id] = chan_inds

    elif method == "threshold":
        peak_values = get_template_amplitudes(
            waveform_extractor, peak_sign=peak_sign, mode="extremum"
        )
        noise = get_noise_levels(
            waveform_extractor.recording, return_scaled=waveform_extractor.return_scaled
        )
        for unit_id in unit_ids:
            chan_inds = np.nonzero((np.abs(peak_values[unit_id]) / noise) >= threshold)
            sparsity_with_index[unit_id] = chan_inds

    elif method == "by_property":
        assert (
            by_property is not None
        ), "Specify the property with the 'by_property' argument!"
        _check_property_consistency(waveform_extractor, by_property)
        rec_by = waveform_extractor.recording.split_by(by_property)
        for unit_id in unit_ids:
            unit_index = waveform_extractor.sorting.id_to_index(unit_id)
            unit_property = waveform_extractor.sorting.get_property(by_property)[
                unit_index
            ]

            assert unit_property in rec_by.keys(), (
                f"Unit property {unit_property} cannot be found in the "
                f"recording properties"
            )
            chan_inds = waveform_extractor.recording.ids_to_indices(
                rec_by[unit_property].get_channel_ids()
            )
            sparsity_with_index[unit_id] = chan_inds

    # handle output ids or indexes
    if outputs == "id":
        sparsity_with_id = {}
        for unit_id in unit_ids:
            chan_inds = sparsity_with_index[unit_id]
            sparsity_with_id[unit_id] = channel_ids[chan_inds]
        return sparsity_with_id
    elif outputs == "index":
        return sparsity_with_index


def get_template_extremum_channel_peak_shift(waveform_extractor, peak_sign: str = "neg"):
    """
    In some situations spike sorters could return a spike index with a small shift related to the waveform peak.
    This function estimates and return these alignment shifts for the mean template.
    This function is internally used by `compute_spike_amplitudes()` to accurately retrieve the spike amplitudes.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')

    Returns
    -------
    shifts: dict
        Dictionary with unit ids as keys and shifts as values
    """
    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids

    extremum_channels_ids = get_template_extremum_channel(
        waveform_extractor, peak_sign=peak_sign
    )

    shifts = {}
    for unit_id in unit_ids:
        chan_id = extremum_channels_ids[unit_id]
        chan_ind = recording.id_to_index(chan_id)

        template = waveform_extractor.get_template(unit_id, mode="average")

        if peak_sign == "both":
            peak_pos = np.argmax(np.abs(template[:, chan_ind]))
        elif peak_sign == "neg":
            peak_pos = np.argmin(template[:, chan_ind])
        elif peak_sign == "pos":
            peak_pos = np.argmax(template[:, chan_ind])
        shift = peak_pos - waveform_extractor.nbefore
        shifts[unit_id] = shift

    return shifts


def get_template_extremum_amplitude(waveform_extractor, peak_sign: str = "neg", mode: str = "at_index"):
    """
    Computes amplitudes on the best channel.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    mode: str
        Where the amplitude is computed
        'extremum':  max or min
        'at_index': take value at spike index

    Returns
    -------
    amplitudes: dict
        Dictionary with unit ids as keys and amplitudes as values
    """
    assert peak_sign in ("both", "neg", "pos")
    assert mode in ("extremum", "at_index")
    unit_ids = waveform_extractor.sorting.unit_ids

    before = waveform_extractor.nbefore

    extremum_channels_ids = get_template_extremum_channel(
        waveform_extractor, peak_sign=peak_sign, mode=mode
    )

    extremum_amplitudes = get_template_amplitudes(
        waveform_extractor, peak_sign=peak_sign, mode=mode
    )

    unit_amplitudes = {}
    for unit_id in unit_ids:
        channel_id = extremum_channels_ids[unit_id]
        best_channel = waveform_extractor.recording.id_to_index(channel_id)
        unit_amplitudes[unit_id] = extremum_amplitudes[unit_id][best_channel]

    return unit_amplitudes


def get_peaks_from_templates(
    waveform_extractor,
    peak_sign="neg",
    radius_um=40,
    **job_kwargs,
):
    """
    Returns a peaks array for each spike using the corresponding template.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    radius_um: float
        Radius in um around the best channel for finding spike max amplitudes.
    Returns
    -------
    peaks: array
        peaks array with 'sample_ind', 'channel_ind', 'amplitude', 'segment_ind'
    """
    from spikeinterface.sortingcomponents.peak_waveform_features import (
        compute_waveform_features_peaks,
    )

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    spikes = sorting.to_spike_vector()
    unit_id_close_channels_inds = get_template_channel_sparsity(
        waveform_extractor,
        peak_sign=peak_sign,
        method="radius",
        radius_um=radius_um,
        outputs="index",
    )
    unit_ids_spikes = np.asarray(
        [sorting.unit_ids[unit_ind] for unit_ind in spikes["unit_ind"]]
    )
    peak_waveform_features = compute_waveform_features_peaks(
        recording,
        spikes,
        time_range_list=[(0.2, 0.2)],
        feature_list=["amplitude"],
        **job_kwargs,
    )
    channel_ind = []
    amplitude = []
    for i, unit_id in enumerate(unit_ids_spikes):
        close_channels_inds = unit_id_close_channels_inds[unit_id]
        abs_amps = np.abs(peak_waveform_features[i, close_channels_inds, 0])
        extremum_channel = close_channels_inds[np.argmax(abs_amps)]
        max_amplitude = peak_waveform_features[i, close_channels_inds, 0][
            np.argmax(abs_amps)
        ]
        channel_ind.append(extremum_channel)
        amplitude.append(max_amplitude)
    channel_ind = np.asarray(channel_ind)
    amplitude = np.asarray(amplitude)
    peaks = rfn.append_fields(
        spikes, ["channel_ind", "amplitude"], [channel_ind, amplitude], usemask=False
    )
    return peaks


def _check_property_consistency(waveform_extractor, by_property):
    assert by_property in waveform_extractor.recording.get_property_keys(), (
        f"Property {by_property} is not a " f"recording property"
    )
    assert by_property in waveform_extractor.sorting.get_property_keys(), (
        f"Property {by_property} is not a " f"sorting property"
    )