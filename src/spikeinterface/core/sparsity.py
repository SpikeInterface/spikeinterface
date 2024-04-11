from __future__ import annotations

import numpy as np


from .basesorting import BaseSorting
from .baserecording import BaseRecording
from .sorting_tools import random_spikes_selection
from .job_tools import _shared_job_kwargs_doc
from .waveform_tools import estimate_templates_with_accumulator


_sparsity_doc = """
    method: str
        * "best_channels": N best channels with the largest amplitude. Use the "num_channels" argument to specify the
                         number of channels.
        * "radius": radius around the best channel. Use the "radius_um" argument to specify the radius in um
        * "snr": threshold based on template signal-to-noise ratio. Use the "threshold" argument
                 to specify the SNR threshold (in units of noise levels)
        * "ptp": threshold based on the peak-to-peak values on every channels. Use the "threshold" argument
                to specify the ptp threshold (in units of noise levels)
        * "energy": threshold based on the expected energy that should be present on the channels,
                    given their noise levels. Use the "threshold" argument to specify the SNR threshold
                    (in units of noise levels)
        * "by_property": sparsity is given by a property of the recording and sorting(e.g. "group").
                         Use the "by_property" argument to specify the property name.

    peak_sign: str
        Sign of the template to compute best channels ("neg", "pos", "both")
    num_channels: int
        Number of channels for "best_channels" method
    radius_um: float
        Radius in um for "radius" method
    threshold: float
        Threshold in SNR "threshold" method
    by_property: object
        Property name for "by_property" method
"""


class ChannelSparsity:
    """
    Handle channel sparsity for a set of units. That is, for every unit,
    it indicates which channels are used to represent the waveform and the rest
    of the non-represented channels are assumed to be zero.

    Internally, sparsity is stored as a boolean mask.

    The ChannelSparsity object can also provide other sparsity representations:

        * ChannelSparsity.unit_id_to_channel_ids : unit_id to channel_ids
        * ChannelSparsity.unit_id_to_channel_indices : unit_id channel_inds

    By default it is constructed with a boolean array:

    >>> sparsity = ChannelSparsity(mask, unit_ids, channel_ids)

    But can be also constructed from a dictionary:

    >>> sparsity = ChannelSparsity.from_unit_id_to_channel_ids(unit_id_to_channel_ids, unit_ids, channel_ids)

    Parameters
    ----------
    mask: np.array of bool
        The sparsity mask (num_units, num_channels)
    unit_ids: list or array
        Unit ids vector or list
    channel_ids: list or array
        Channel ids vector or list

    Examples
    --------

    The class can also be used to construct/estimate the sparsity from a SortingAnalyzer or a Templates
    with several methods:

    Using the N best channels (largest template amplitude):

    >>> sparsity = ChannelSparsity.from_best_channels(sorting_analyzer, num_channels, peak_sign="neg")

    Using a neighborhood by radius:

    >>> sparsity = ChannelSparsity.from_radius(sorting_analyzer, radius_um, peak_sign="neg")

    Using a SNR threshold:
    >>> sparsity = ChannelSparsity.from_snr(sorting_analyzer, threshold, peak_sign="neg")

    Using a template energy threshold:
    >>> sparsity = ChannelSparsity.from_energy(sorting_analyzer, threshold)

    Using a recording/sorting property (e.g. "group"):

    >>> sparsity = ChannelSparsity.from_property(sorting_analyzer, by_property="group")

    """

    def __init__(self, mask, unit_ids, channel_ids):
        self.unit_ids = np.asarray(unit_ids)
        self.channel_ids = np.asarray(channel_ids)
        self.mask = np.asarray(mask, dtype="bool")
        assert self.mask.shape[0] == self.unit_ids.shape[0]
        assert self.mask.shape[1] == self.channel_ids.shape[0]

        # Those are computed at first call
        self._unit_id_to_channel_ids = None
        self._unit_id_to_channel_indices = None

        self.num_channels = self.channel_ids.size
        self.num_units = self.unit_ids.size
        if self.mask.shape[0]:
            self.max_num_active_channels = self.mask.sum(axis=1).max()
        else:
            # empty sorting without units
            self.max_num_active_channels = 0

    def __repr__(self):
        density = np.mean(self.mask)
        txt = f"ChannelSparsity - units: {self.num_units} - channels: {self.num_channels} - density, P(x=1): {density:0.2f}"
        return txt

    @property
    def unit_id_to_channel_ids(self):
        if self._unit_id_to_channel_ids is None:
            self._unit_id_to_channel_ids = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._unit_id_to_channel_ids[unit_id] = self.channel_ids[channel_inds]
        return self._unit_id_to_channel_ids

    @property
    def unit_id_to_channel_indices(self):
        if self._unit_id_to_channel_indices is None:
            self._unit_id_to_channel_indices = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._unit_id_to_channel_indices[unit_id] = channel_inds
        return self._unit_id_to_channel_indices

    def sparsify_waveforms(self, waveforms: np.ndarray, unit_id: str | int) -> np.ndarray:
        """
        Sparsify the waveforms according to a unit_id corresponding sparsity.


        Given a unit_id, this method selects only the active channels for
        that unit and removes the rest.

        Parameters
        ----------
        waveforms : np.array
            Dense waveforms with shape (num_waveforms, num_samples, num_channels) or a
            single dense waveform (template) with shape (num_samples, num_channels).
        unit_id : str
            The unit_id for which to sparsify the waveform.

        Returns
        -------
        sparsified_waveforms : np.array
            Sparse waveforms with shape (num_waveforms, num_samples, num_active_channels)
            or a single sparsified waveform (template) with shape (num_samples, num_active_channels).
        """

        non_zero_indices = self.unit_id_to_channel_indices[unit_id]
        sparsified_waveforms = waveforms[..., non_zero_indices]

        return sparsified_waveforms

    def densify_waveforms(self, waveforms: np.ndarray, unit_id: str | int) -> np.ndarray:
        """
        Densify sparse waveforms that were sparisified according to a unit's channel sparsity.

        Given a unit_id its sparsified waveform, this method places the waveform back
        into its original form within a dense array.

        Parameters
        ----------
        waveforms : np.array
            The sparsified waveforms array of shape (num_waveforms, num_samples, num_active_channels) or a single
            sparse waveform (template) with shape (num_samples, num_active_channels).
        unit_id : str
            The unit_id that was used to sparsify the waveform.

        Returns
        -------
        densified_waveforms : np.array
            The densified waveforms array of shape (num_waveforms, num_samples, num_channels) or a single dense
            waveform (template) with shape (num_samples, num_channels).

        """

        non_zero_indices = self.unit_id_to_channel_indices[unit_id]
        num_active_channels = len(non_zero_indices)

        if not self.are_waveforms_sparse(waveforms=waveforms, unit_id=unit_id):
            error_message = (
                "Waveforms do not seem to be in the sparsity shape for this unit_id. The number of active channels is "
                f"{num_active_channels}, but the waveform has non-zero values outsies of those active channels: \n"
                f"{waveforms[..., num_active_channels:]}"
            )
            raise ValueError(error_message)

        densified_shape = waveforms.shape[:-1] + (self.num_channels,)
        densified_waveforms = np.zeros(shape=densified_shape, dtype=waveforms.dtype)
        # Maps the active channels to their original indices
        densified_waveforms[..., non_zero_indices] = waveforms[..., :num_active_channels]

        return densified_waveforms

    def are_waveforms_dense(self, waveforms: np.ndarray) -> bool:
        return waveforms.shape[-1] == self.num_channels

    def are_waveforms_sparse(self, waveforms: np.ndarray, unit_id: str | int) -> bool:
        non_zero_indices = self.unit_id_to_channel_indices[unit_id]
        num_active_channels = len(non_zero_indices)

        # If any channel is non-zero outside of the active channels, then the waveforms are not sparse
        excess_zeros = waveforms[..., num_active_channels:].sum()

        return int(excess_zeros) == 0

    def sparsify_templates(self, templates_array: np.ndarray) -> np.ndarray:
        assert templates_array.shape[0] == self.num_units
        assert templates_array.shape[2] == self.num_channels

        max_num_active_channels = self.max_num_active_channels
        sparsified_shape = (self.num_units, templates_array.shape[1], max_num_active_channels)
        sparse_templates = np.zeros(shape=sparsified_shape, dtype=templates_array.dtype)
        for unit_index, unit_id in enumerate(self.unit_ids):
            template = templates_array[unit_index, ...]
            sparse_template = self.sparsify_waveforms(waveforms=template[np.newaxis, :, :], unit_id=unit_id)
            sparse_templates[unit_index, :, : sparse_template.shape[2]] = sparse_template

        return sparse_templates

    @classmethod
    def from_unit_id_to_channel_ids(cls, unit_id_to_channel_ids, unit_ids, channel_ids):
        """
        Create a sparsity object from dict unit_id to channel_ids.
        """
        unit_ids = list(unit_ids)
        channel_ids = list(channel_ids)
        mask = np.zeros((len(unit_ids), len(channel_ids)), dtype="bool")
        for unit_id, chan_ids in unit_id_to_channel_ids.items():
            unit_ind = unit_ids.index(unit_id)
            channel_inds = [channel_ids.index(chan_id) for chan_id in chan_ids]
            mask[unit_ind, channel_inds] = True
        return cls(mask, unit_ids, channel_ids)

    def to_dict(self):
        """
        Return a serializable dict.
        """
        return dict(
            unit_id_to_channel_ids={k: list(v) for k, v in self.unit_id_to_channel_ids.items()},
            channel_ids=list(self.channel_ids),
            unit_ids=list(self.unit_ids),
        )

    @classmethod
    def from_dict(cls, dictionary: dict):
        unit_id_to_channel_ids_corrected = {}
        for unit_id in dictionary["unit_ids"]:
            if unit_id in dictionary["unit_id_to_channel_ids"]:
                unit_id_to_channel_ids_corrected[unit_id] = dictionary["unit_id_to_channel_ids"][unit_id]
            else:
                unit_id_to_channel_ids_corrected[unit_id] = dictionary["unit_id_to_channel_ids"][str(unit_id)]
        dictionary["unit_id_to_channel_ids"] = unit_id_to_channel_ids_corrected

        return cls.from_unit_id_to_channel_ids(**dictionary)

    ## Some convinient function to compute sparsity from several strategy
    @classmethod
    def from_best_channels(cls, templates_or_sorting_analyzer, num_channels, peak_sign="neg"):
        """
        Construct sparsity from N best channels with the largest amplitude.
        Use the "num_channels" argument to specify the number of channels.
        """
        from .template_tools import get_template_amplitudes

        print(templates_or_sorting_analyzer)
        mask = np.zeros(
            (templates_or_sorting_analyzer.unit_ids.size, templates_or_sorting_analyzer.channel_ids.size), dtype="bool"
        )
        peak_values = get_template_amplitudes(templates_or_sorting_analyzer, peak_sign=peak_sign)
        for unit_ind, unit_id in enumerate(templates_or_sorting_analyzer.unit_ids):
            chan_inds = np.argsort(np.abs(peak_values[unit_id]))[::-1]
            chan_inds = chan_inds[:num_channels]
            mask[unit_ind, chan_inds] = True
        return cls(mask, templates_or_sorting_analyzer.unit_ids, templates_or_sorting_analyzer.channel_ids)

    @classmethod
    def from_radius(cls, templates_or_sorting_analyzer, radius_um, peak_sign="neg"):
        """
        Construct sparsity from a radius around the best channel.
        Use the "radius_um" argument to specify the radius in um
        """
        from .template_tools import get_template_extremum_channel

        mask = np.zeros(
            (templates_or_sorting_analyzer.unit_ids.size, templates_or_sorting_analyzer.channel_ids.size), dtype="bool"
        )
        channel_locations = templates_or_sorting_analyzer.get_channel_locations()
        distances = np.linalg.norm(channel_locations[:, np.newaxis] - channel_locations[np.newaxis, :], axis=2)
        best_chan = get_template_extremum_channel(templates_or_sorting_analyzer, peak_sign=peak_sign, outputs="index")
        for unit_ind, unit_id in enumerate(templates_or_sorting_analyzer.unit_ids):
            chan_ind = best_chan[unit_id]
            (chan_inds,) = np.nonzero(distances[chan_ind, :] <= radius_um)
            mask[unit_ind, chan_inds] = True
        return cls(mask, templates_or_sorting_analyzer.unit_ids, templates_or_sorting_analyzer.channel_ids)

    @classmethod
    def from_snr(cls, templates_or_sorting_analyzer, threshold, noise_levels=None, peak_sign="neg"):
        """
        Construct sparsity from a thresholds based on template signal-to-noise ratio.
        Use the "threshold" argument to specify the SNR threshold.
        """
        from .template_tools import get_template_amplitudes
        from .sortinganalyzer import SortingAnalyzer
        from .template import Templates

        assert (
            templates_or_sorting_analyzer.sparsity is None
        ), "To compute sparsity you need a dense SortingAnalyzer or Templates"

        unit_ids = templates_or_sorting_analyzer.unit_ids
        channel_ids = templates_or_sorting_analyzer.channel_ids

        if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
            ext = templates_or_sorting_analyzer.get_extension("noise_levels")
            assert ext is not None, "To compute sparsity from snr you need to compute 'noise_levels' first"
            noise_levels = ext.data["noise_levels"]
            return_scaled = templates_or_sorting_analyzer.return_scaled
        elif isinstance(templates_or_sorting_analyzer, Templates):
            assert noise_levels is not None
            return_scaled = True

        mask = np.zeros((unit_ids.size, channel_ids.size), dtype="bool")

        peak_values = get_template_amplitudes(
            templates_or_sorting_analyzer, peak_sign=peak_sign, mode="extremum", return_scaled=return_scaled
        )

        for unit_ind, unit_id in enumerate(unit_ids):
            chan_inds = np.nonzero((np.abs(peak_values[unit_id]) / noise_levels) >= threshold)
            mask[unit_ind, chan_inds] = True
        return cls(mask, unit_ids, channel_ids)

    @classmethod
    def from_ptp(cls, templates_or_sorting_analyzer, threshold, noise_levels=None):
        """
        Construct sparsity from a thresholds based on template peak-to-peak values.
        Use the "threshold" argument to specify the SNR threshold.
        """

        assert (
            templates_or_sorting_analyzer.sparsity is None
        ), "To compute sparsity you need a dense SortingAnalyzer or Templates"

        from .template_tools import get_template_amplitudes
        from .sortinganalyzer import SortingAnalyzer
        from .template import Templates

        unit_ids = templates_or_sorting_analyzer.unit_ids
        channel_ids = templates_or_sorting_analyzer.channel_ids

        if isinstance(templates_or_sorting_analyzer, SortingAnalyzer):
            ext = templates_or_sorting_analyzer.get_extension("noise_levels")
            assert ext is not None, "To compute sparsity from snr you need to compute 'noise_levels' first"
            noise_levels = ext.data["noise_levels"]
            return_scaled = templates_or_sorting_analyzer.return_scaled
        elif isinstance(templates_or_sorting_analyzer, Templates):
            assert noise_levels is not None
            return_scaled = True

        from .template_tools import get_dense_templates_array

        mask = np.zeros((unit_ids.size, channel_ids.size), dtype="bool")

        templates_array = get_dense_templates_array(templates_or_sorting_analyzer, return_scaled=return_scaled)
        templates_ptps = np.ptp(templates_array, axis=1)

        for unit_ind, unit_id in enumerate(unit_ids):
            chan_inds = np.nonzero(templates_ptps[unit_ind] / noise_levels >= threshold)
            mask[unit_ind, chan_inds] = True
        return cls(mask, unit_ids, channel_ids)

    @classmethod
    def from_energy(cls, sorting_analyzer, threshold):
        """
        Construct sparsity from a threshold based on per channel energy ratio.
        Use the "threshold" argument to specify the SNR threshold.
        """
        assert sorting_analyzer.sparsity is None, "To compute sparsity with energy you need a dense SortingAnalyzer"

        mask = np.zeros((sorting_analyzer.unit_ids.size, sorting_analyzer.channel_ids.size), dtype="bool")

        # noise_levels
        ext = sorting_analyzer.get_extension("noise_levels")
        assert ext is not None, "To compute sparsity from ptp you need to compute 'noise_levels' first"
        noise_levels = ext.data["noise_levels"]

        # waveforms
        ext_waveforms = sorting_analyzer.get_extension("waveforms")
        assert ext_waveforms is not None, "To compute sparsity from energy you need to compute 'waveforms' first"
        namples = ext_waveforms.nbefore + ext_waveforms.nafter

        noise = np.sqrt(namples) * noise_levels

        for unit_ind, unit_id in enumerate(sorting_analyzer.unit_ids):
            wfs = ext_waveforms.get_waveforms_one_unit(unit_id, force_dense=True)
            energies = np.linalg.norm(wfs, axis=(0, 1))
            chan_inds = np.nonzero(energies / (noise * np.sqrt(len(wfs))) >= threshold)
            mask[unit_ind, chan_inds] = True
        return cls(mask, sorting_analyzer.unit_ids, sorting_analyzer.channel_ids)

    @classmethod
    def from_property(cls, sorting_analyzer, by_property):
        """
        Construct sparsity witha property of the recording and sorting(e.g. "group").
        Use the "by_property" argument to specify the property name.
        """
        # check consistency
        assert (
            by_property in sorting_analyzer.recording.get_property_keys()
        ), f"Property {by_property} is not a recording property"
        assert (
            by_property in sorting_analyzer.sorting.get_property_keys()
        ), f"Property {by_property} is not a sorting property"

        mask = np.zeros((sorting_analyzer.unit_ids.size, sorting_analyzer.channel_ids.size), dtype="bool")
        rec_by = sorting_analyzer.recording.split_by(by_property)
        for unit_ind, unit_id in enumerate(sorting_analyzer.unit_ids):
            unit_property = sorting_analyzer.sorting.get_property(by_property)[unit_ind]
            assert (
                unit_property in rec_by.keys()
            ), f"Unit property {unit_property} cannot be found in the recording properties"
            chan_inds = sorting_analyzer.recording.ids_to_indices(rec_by[unit_property].get_channel_ids())
            mask[unit_ind, chan_inds] = True
        return cls(mask, sorting_analyzer.unit_ids, sorting_analyzer.channel_ids)

    @classmethod
    def create_dense(cls, sorting_analyzer):
        """
        Create a sparsity object with all selected channel for all units.
        """
        mask = np.ones((sorting_analyzer.unit_ids.size, sorting_analyzer.channel_ids.size), dtype="bool")
        return cls(mask, sorting_analyzer.unit_ids, sorting_analyzer.channel_ids)


def compute_sparsity(
    templates_or_sorting_analyzer,
    noise_levels=None,
    method="radius",
    peak_sign="neg",
    num_channels=5,
    radius_um=100.0,
    threshold=5,
    by_property=None,
):
    """
    Get channel sparsity (subset of channels) for each template with several methods.

    Parameters
    ----------
    templates_or_sorting_analyzer: Templates | SortingAnalyzer
        A Templates or a SortingAnalyzer object.
        Some methods accept both objects ("best_channels", "radius", )
        Other methods require only SortingAnalyzer because internally the recording is needed.

    {}

    Returns
    -------
    sparsity: ChannelSparsity
        The estimated sparsity
    """

    # Can't be done at module because this is a cyclic import, too bad
    from .template import Templates
    from .waveforms_extractor_backwards_compatibility import MockWaveformExtractor
    from .sortinganalyzer import SortingAnalyzer

    if isinstance(templates_or_sorting_analyzer, MockWaveformExtractor):
        # to keep backward compatibility
        templates_or_sorting_analyzer = templates_or_sorting_analyzer.sorting_analyzer

    if method in ("best_channels", "radius", "snr", "ptp"):
        assert isinstance(
            templates_or_sorting_analyzer, (Templates, SortingAnalyzer)
        ), f"compute_sparsity(method='{method}') need Templates or SortingAnalyzer"
    else:
        assert isinstance(
            templates_or_sorting_analyzer, SortingAnalyzer
        ), f"compute_sparsity(method='{method}') need SortingAnalyzer"

    if method in ("snr", "ptp") and isinstance(templates_or_sorting_analyzer, Templates):
        assert (
            noise_levels is not None
        ), f"compute_sparsity(..., method='{method}') with Templates need noise_levels as input"

    if method == "best_channels":
        assert num_channels is not None, "For the 'best_channels' method, 'num_channels' needs to be given"
        sparsity = ChannelSparsity.from_best_channels(templates_or_sorting_analyzer, num_channels, peak_sign=peak_sign)
    elif method == "radius":
        assert radius_um is not None, "For the 'radius' method, 'radius_um' needs to be given"
        sparsity = ChannelSparsity.from_radius(templates_or_sorting_analyzer, radius_um, peak_sign=peak_sign)
    elif method == "snr":
        assert threshold is not None, "For the 'snr' method, 'threshold' needs to be given"
        sparsity = ChannelSparsity.from_snr(
            templates_or_sorting_analyzer, threshold, noise_levels=noise_levels, peak_sign=peak_sign
        )
    elif method == "ptp":
        assert threshold is not None, "For the 'ptp' method, 'threshold' needs to be given"
        sparsity = ChannelSparsity.from_ptp(
            templates_or_sorting_analyzer,
            threshold,
            noise_levels=noise_levels,
        )
    elif method == "energy":
        assert threshold is not None, "For the 'energy' method, 'threshold' needs to be given"
        sparsity = ChannelSparsity.from_energy(templates_or_sorting_analyzer, threshold)
    elif method == "by_property":
        assert by_property is not None, "For the 'by_property' method, 'by_property' needs to be given"
        sparsity = ChannelSparsity.from_property(templates_or_sorting_analyzer, by_property)
    else:
        raise ValueError(f"compute_sparsity() method={method} does not exists")

    return sparsity


compute_sparsity.__doc__ = compute_sparsity.__doc__.format(_sparsity_doc)


def estimate_sparsity(
    recording: BaseRecording,
    sorting: BaseSorting,
    num_spikes_for_sparsity: int = 100,
    ms_before: float = 1.0,
    ms_after: float = 2.5,
    method: "radius" | "best_channels" = "radius",
    peak_sign: str = "neg",
    radius_um: float = 100.0,
    num_channels: int = 5,
    **job_kwargs,
):
    """
    Estimate the sparsity without needing a SortingAnalyzer or Templates object
    This is faster than  `spikeinterface.waveforms_extractor.precompute_sparsity()` and it
    traverses the recording to compute the average templates for each unit.

    Contrary to the previous implementation:
      * all units are computed in one read of recording
      * it doesn't require a folder
      * it doesn't consume too much memory
      * it uses internally the `estimate_templates_with_accumulator()` which is fast and parallel

    Parameters
    ----------
    recording: BaseRecording
        The recording
    sorting: BaseSorting
        The sorting
    num_spikes_for_sparsity: int, default: 100
        How many spikes per units to compute the sparsity
    ms_before: float, default: 1.0
        Cut out in ms before spike time
    ms_after: float, default: 2.5
        Cut out in ms after spike time
    method: "radius" | "best_channels", default: "radius"
        Sparsity method propagated to the `compute_sparsity()` function.
        Only "radius" or "best_channels" are implemented
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels
    radius_um: float, default: 100.0
        Used for "radius" method
    num_channels: int, default: 5
        Used for "best_channels" method

    {}

    Returns
    -------
    sparsity: ChannelSparsity
        The estimated sparsity
    """
    # Can't be done at module because this is a cyclic import, too bad
    from .template import Templates

    assert method in ("radius", "best_channels"), "estimate_sparsity() handle only method='radius' or 'best_channel'"

    if recording.get_probes() == 1:
        # standard case
        probe = recording.get_probe()
    else:
        # if many probe or no probe then we use channel location and create a dummy probe with all channels
        # note that get_channel_locations() is checking that channel are not spatialy overlapping so the radius method is OK.
        chan_locs = recording.get_channel_locations()
        probe = recording.create_dummy_probe_from_locations(chan_locs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]
    random_spikes_indices = random_spikes_selection(
        sorting,
        num_samples,
        method="uniform",
        max_spikes_per_unit=num_spikes_for_sparsity,
        margin_size=max(nbefore, nafter),
        seed=2205,
    )
    spikes = sorting.to_spike_vector()
    spikes = spikes[random_spikes_indices]

    templates_array = estimate_templates_with_accumulator(
        recording,
        spikes,
        sorting.unit_ids,
        nbefore,
        nafter,
        return_scaled=False,
        job_name="estimate_sparsity",
        **job_kwargs,
    )
    templates = Templates(
        templates_array=templates_array,
        sampling_frequency=recording.sampling_frequency,
        nbefore=nbefore,
        sparsity_mask=None,
        channel_ids=recording.channel_ids,
        unit_ids=sorting.unit_ids,
        probe=probe,
    )

    sparsity = compute_sparsity(
        templates, method=method, peak_sign=peak_sign, num_channels=num_channels, radius_um=radius_um
    )

    return sparsity


estimate_sparsity.__doc__ = estimate_sparsity.__doc__.format(_shared_job_kwargs_doc)
