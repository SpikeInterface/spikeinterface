from __future__ import annotations

import numpy as np

from spikeinterface.core import ChannelSparsity, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


# DEBUG = True


class AmplitudeScalingsCalculator(BaseWaveformExtractorExtension):
    """
    Computes amplitude scalings from WaveformExtractor.
    """

    extension_name = "amplitude_scalings"
    handle_sparsity = True

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(
            extremum_channel_inds=extremum_channel_inds, use_cache=False
        )
        self.collisions = None

    def _set_params(
        self,
        sparsity,
        max_dense_channels,
        ms_before,
        ms_after,
        handle_collisions,
        delta_collision_ms,
    ):
        params = dict(
            sparsity=sparsity,
            max_dense_channels=max_dense_channels,
            ms_before=ms_before,
            ms_after=ms_after,
            handle_collisions=handle_collisions,
            delta_collision_ms=delta_collision_ms,
        )
        return params

    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.isin(old_unit_ids, unit_ids))

        spike_mask = np.isin(self.spikes["unit_index"], unit_inds)
        new_amplitude_scalings = self._extension_data["amplitude_scalings"][spike_mask]
        return dict(amplitude_scalings=new_amplitude_scalings)

    def _run(self, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        we = self.waveform_extractor
        recording = we.recording
        nbefore = we.nbefore
        nafter = we.nafter
        ms_before = self._params["ms_before"]
        ms_after = self._params["ms_after"]

        # collisions
        handle_collisions = self._params["handle_collisions"]
        delta_collision_ms = self._params["delta_collision_ms"]
        delta_collision_samples = int(delta_collision_ms / 1000 * we.sampling_frequency)

        return_scaled = we._params["return_scaled"]

        if ms_before is not None:
            assert (
                ms_before <= we._params["ms_before"]
            ), f"`ms_before` must be smaller than `ms_before` used in WaveformExractor: {we._params['ms_before']}"
        if ms_after is not None:
            assert (
                ms_after <= we._params["ms_after"]
            ), f"`ms_after` must be smaller than `ms_after` used in WaveformExractor: {we._params['ms_after']}"

        cut_out_before = int(ms_before / 1000 * we.sampling_frequency) if ms_before is not None else nbefore
        cut_out_after = int(ms_after / 1000 * we.sampling_frequency) if ms_after is not None else nafter

        if we.is_sparse() and self._params["sparsity"] is None:
            sparsity = we.sparsity
        elif we.is_sparse() and self._params["sparsity"] is not None:
            sparsity = self._params["sparsity"]
            # assert provided sparsity is sparser than the one in the waveform extractor
            waveform_sparsity = we.sparsity
            assert np.all(
                np.sum(waveform_sparsity.mask, 1) - np.sum(sparsity.mask, 1) > 0
            ), "The provided sparsity needs to be sparser than the one in the waveform extractor!"
        elif not we.is_sparse() and self._params["sparsity"] is not None:
            sparsity = self._params["sparsity"]
        else:
            if self._params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self._params["max_dense_channels"], ""
            sparsity = ChannelSparsity.create_dense(we)
        sparsity_mask = sparsity.mask
        all_templates = we.get_all_templates()

        # precompute segment slice
        segment_slices = []
        for segment_index in range(we.get_num_segments()):
            i0, i1 = np.searchsorted(self.spikes["segment_index"], [segment_index, segment_index + 1])
            segment_slices.append(slice(i0, i1))

        # and run
        func = _amplitude_scalings_chunk
        init_func = _init_worker_amplitude_scalings
        n_jobs = ensure_n_jobs(recording, job_kwargs.get("n_jobs", None))
        job_kwargs["n_jobs"] = n_jobs
        init_args = (
            recording,
            self.spikes,
            all_templates,
            segment_slices,
            sparsity_mask,
            nbefore,
            nafter,
            cut_out_before,
            cut_out_after,
            return_scaled,
            handle_collisions,
            delta_collision_samples,
        )
        processor = ChunkRecordingExecutor(
            recording,
            func,
            init_func,
            init_args,
            handle_returns=True,
            job_name="extract amplitude scalings",
            **job_kwargs,
        )
        out = processor.run()
        (amp_scalings, collisions) = zip(*out)
        amp_scalings = np.concatenate(amp_scalings)

        collisions_dict = {}
        if handle_collisions:
            for collision in collisions:
                collisions_dict.update(collision)
            self.collisions = collisions_dict
            # Note: collisions are note in _extension_data because they are not pickable. We only store the indices
            self._extension_data["collisions"] = np.array(list(collisions_dict.keys()))

        self._extension_data["amplitude_scalings"] = amp_scalings

    def get_data(self, outputs="concatenated"):
        """
        Get computed spike amplitudes.
        Parameters
        ----------
        outputs : "concatenated" | "by_unit", default: "concatenated"
            The output format

        Returns
        -------
        spike_amplitudes : np.array or dict
            The spike amplitudes as an array (outputs="concatenated") or
            as a dict with units as key and spike amplitudes as values.
        """
        we = self.waveform_extractor
        sorting = we.sorting

        if outputs == "concatenated":
            return self._extension_data[f"amplitude_scalings"]
        elif outputs == "by_unit":
            amplitudes_by_unit = []
            for segment_index in range(we.get_num_segments()):
                amplitudes_by_unit.append({})
                segment_mask = self.spikes["segment_index"] == segment_index
                spikes_segment = self.spikes[segment_mask]
                amp_scalings_segment = self._extension_data[f"amplitude_scalings"][segment_mask]
                for unit_index, unit_id in enumerate(sorting.unit_ids):
                    unit_mask = spikes_segment["unit_index"] == unit_index
                    amp_scalings = amp_scalings_segment[unit_mask]
                    amplitudes_by_unit[segment_index][unit_id] = amp_scalings
            return amplitudes_by_unit

    @staticmethod
    def get_extension_function():
        return compute_amplitude_scalings


WaveformExtractor.register_extension(AmplitudeScalingsCalculator)


def compute_amplitude_scalings(
    waveform_extractor,
    sparsity=None,
    max_dense_channels=16,
    ms_before=None,
    ms_after=None,
    handle_collisions=True,
    delta_collision_ms=2,
    load_if_exists=False,
    outputs="concatenated",
    **job_kwargs,
):
    """
    Computes the amplitude scalings from a WaveformExtractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    sparsity: ChannelSparsity or None, default: None
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
    max_dense_channels: int, default: 16
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
    ms_before : float or None, default: None
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_before is used.
    ms_after : float or None, default: None
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_after is used.
    handle_collisions: bool, default: True
        Whether to handle collisions between spikes. If True, the amplitude scaling of colliding spikes
        (defined as spikes within `delta_collision_ms` ms and with overlapping sparsity) is computed by fitting a
        multi-linear regression model (with `sklearn.LinearRegression`). If False, each spike is fitted independently.
    delta_collision_ms: float, default: 2
        The maximum time difference in ms before and after a spike to gather colliding spikes.
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    outputs: "concatenated" | "by_unit", default: "concatenated"
        How the output should be returned
    {}

    Returns
    -------
    amplitude_scalings: np.array or list of dict
        The amplitude scalings.
            - If "concatenated" all amplitudes for all spikes and all units are concatenated
            - If "by_unit", amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(AmplitudeScalingsCalculator.extension_name):
        sac = waveform_extractor.load_extension(AmplitudeScalingsCalculator.extension_name)
    else:
        sac = AmplitudeScalingsCalculator(waveform_extractor)
        sac.set_params(
            sparsity=sparsity,
            max_dense_channels=max_dense_channels,
            ms_before=ms_before,
            ms_after=ms_after,
            handle_collisions=handle_collisions,
            delta_collision_ms=delta_collision_ms,
        )
        sac.run(**job_kwargs)

    amps = sac.get_data(outputs=outputs)
    return amps


compute_amplitude_scalings.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_amplitude_scalings(
    recording,
    spikes,
    all_templates,
    segment_slices,
    sparsity_mask,
    nbefore,
    nafter,
    cut_out_before,
    cut_out_after,
    return_scaled,
    handle_collisions,
    delta_collision_samples,
):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["spikes"] = spikes
    worker_ctx["all_templates"] = all_templates
    worker_ctx["segment_slices"] = segment_slices
    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["cut_out_before"] = cut_out_before
    worker_ctx["cut_out_after"] = cut_out_after
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["sparsity_mask"] = sparsity_mask
    worker_ctx["handle_collisions"] = handle_collisions
    worker_ctx["delta_collision_samples"] = delta_collision_samples

    if not handle_collisions:
        worker_ctx["margin"] = max(nbefore, nafter)
    else:
        # in this case we extend the margin to be able to get with collisions outside the chunk
        margin_waveforms = max(nbefore, nafter)
        max_margin_collisions = delta_collision_samples + margin_waveforms
        worker_ctx["margin"] = max_margin_collisions

    return worker_ctx


def _amplitude_scalings_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # from sklearn.linear_model import LinearRegression
    from scipy.stats import linregress

    # recover variables of the worker
    spikes = worker_ctx["spikes"]
    recording = worker_ctx["recording"]
    all_templates = worker_ctx["all_templates"]
    segment_slices = worker_ctx["segment_slices"]
    sparsity_mask = worker_ctx["sparsity_mask"]
    nbefore = worker_ctx["nbefore"]
    cut_out_before = worker_ctx["cut_out_before"]
    cut_out_after = worker_ctx["cut_out_after"]
    margin = worker_ctx["margin"]
    return_scaled = worker_ctx["return_scaled"]
    handle_collisions = worker_ctx["handle_collisions"]
    delta_collision_samples = worker_ctx["delta_collision_samples"]

    spikes_in_segment = spikes[segment_slices[segment_index]]

    i0, i1 = np.searchsorted(spikes_in_segment["sample_index"], [start_frame, end_frame])

    if i0 != i1:
        local_spikes = spikes_in_segment[i0:i1]
        traces_with_margin, left, right = get_chunk_with_margin(
            recording._recording_segments[segment_index], start_frame, end_frame, channel_indices=None, margin=margin
        )

        # scale traces with margin to match scaling of templates
        if return_scaled and recording.has_scaled():
            gains = recording.get_property("gain_to_uV")
            offsets = recording.get_property("offset_to_uV")
            traces_with_margin = traces_with_margin.astype("float32") * gains + offsets

        # set colliding spikes apart (if needed)
        if handle_collisions:
            # local spikes with margin!
            i0_margin, i1_margin = np.searchsorted(
                spikes_in_segment["sample_index"], [start_frame - left, end_frame + right]
            )
            local_spikes_w_margin = spikes_in_segment[i0_margin:i1_margin]
            collisions_local = find_collisions(
                local_spikes, local_spikes_w_margin, delta_collision_samples, sparsity_mask
            )
        else:
            collisions_local = {}

        # compute the scaling for each spike
        scalings = np.zeros(len(local_spikes), dtype=float)
        # collision_global transforms local spike index to global spike index
        collisions_global = {}
        for spike_index, spike in enumerate(local_spikes):
            if spike_index in collisions_local.keys():
                # we deal with overlapping spikes later
                continue
            unit_index = spike["unit_index"]
            sample_index = spike["sample_index"]
            (sparse_indices,) = np.nonzero(sparsity_mask[unit_index])
            template = all_templates[unit_index][:, sparse_indices]
            template = template[nbefore - cut_out_before : nbefore + cut_out_after]
            sample_centered = sample_index - start_frame
            cut_out_start = left + sample_centered - cut_out_before
            cut_out_end = left + sample_centered + cut_out_after
            if sample_index - cut_out_before < 0:
                local_waveform = traces_with_margin[:cut_out_end, sparse_indices]
                template = template[cut_out_before - sample_index :]
            elif sample_index + cut_out_after > end_frame + right:
                local_waveform = traces_with_margin[cut_out_start:, sparse_indices]
                template = template[: -(sample_index + cut_out_after - (end_frame + right))]
            else:
                local_waveform = traces_with_margin[cut_out_start:cut_out_end, sparse_indices]
            assert template.shape == local_waveform.shape

            # here we use linregress, which is equivalent to using sklearn LinearRegression with fit_intercept=True
            # y = local_waveform.flatten()
            # X = template.flatten()[:, np.newaxis]
            # reg = LinearRegression(positive=True, fit_intercept=True).fit(X, y)
            # scalings[spike_index] = reg.coef_[0]
            linregress_res = linregress(template.flatten(), local_waveform.flatten())
            scalings[spike_index] = linregress_res[0]

        # deal with collisions
        if len(collisions_local) > 0:
            num_spikes_in_previous_segments = int(
                np.sum([len(spikes[segment_slices[s]]) for s in range(segment_index)])
            )
            for spike_index, collision in collisions_local.items():
                scaled_amps = fit_collision(
                    collision,
                    traces_with_margin,
                    start_frame,
                    end_frame,
                    left,
                    right,
                    nbefore,
                    all_templates,
                    sparsity_mask,
                    cut_out_before,
                    cut_out_after,
                )
                # the scaling for the current spike is at index 0
                scalings[spike_index] = scaled_amps[0]

                # make collision_dict indices "absolute" by adding i0 and the cumulative number of spikes in previous segments
                collisions_global.update({spike_index + i0 + num_spikes_in_previous_segments: collision})
    else:
        scalings = np.array([])
        collisions_global = {}

    return (scalings, collisions_global)


### Collision handling ###
def _are_unit_indices_overlapping(sparsity_mask, i, j):
    """
    Returns True if the unit indices i and j are overlapping, False otherwise

    Parameters
    ----------
    sparsity_mask: boolean mask
        The sparsity mask
    i: int
        The first unit index
    j: int
        The second unit index

    Returns
    -------
    bool
        True if the unit indices i and j are overlapping, False otherwise
    """
    if np.any(sparsity_mask[i] & sparsity_mask[j]):
        return True
    else:
        return False


def find_collisions(spikes, spikes_w_margin, delta_collision_samples, sparsity_mask):
    """
    Finds the collisions between spikes.

    Parameters
    ----------
    spikes: np.array
        An array of spikes
    spikes_w_margin: np.array
        An array of spikes within the added margin
    delta_collision_samples: int
        The maximum number of samples between two spikes to consider them as overlapping
    sparsity_mask: boolean mask
        The sparsity mask

    Returns
    -------
    collision_spikes_dict: np.array
        A dictionary with collisions. The key is the index of the spike with collision, the value is an
        array of overlapping spikes, including the spike itself at position 0.
    """
    # TODO: refactor to speed-up
    collision_spikes_dict = {}
    for spike_index, spike in enumerate(spikes):
        # find the index of the spike within the spikes_w_margin
        spike_index_w_margin = np.where(spikes_w_margin == spike)[0][0]

        # find the possible spikes per and post within delta_collision_samples
        consecutive_window_pre, consecutive_window_post = np.searchsorted(
            spikes_w_margin["sample_index"],
            [spike["sample_index"] - delta_collision_samples, spike["sample_index"] + delta_collision_samples],
        )

        # exclude the spike itself (it is included in the collision_spikes by construction)
        pre_possible_consecutive_spike_indices = np.arange(consecutive_window_pre, spike_index_w_margin)
        post_possible_consecutive_spike_indices = np.arange(spike_index_w_margin + 1, consecutive_window_post)
        possible_overlapping_spike_indices = np.concatenate(
            (pre_possible_consecutive_spike_indices, post_possible_consecutive_spike_indices)
        )

        # find the overlapping spikes in space as well
        for possible_overlapping_spike_index in possible_overlapping_spike_indices:
            if _are_unit_indices_overlapping(
                sparsity_mask,
                spike["unit_index"],
                spikes_w_margin[possible_overlapping_spike_index]["unit_index"],
            ):
                if spike_index not in collision_spikes_dict:
                    collision_spikes_dict[spike_index] = np.array([spike])
                collision_spikes_dict[spike_index] = np.concatenate(
                    (collision_spikes_dict[spike_index], [spikes_w_margin[possible_overlapping_spike_index]])
                )
    return collision_spikes_dict


def fit_collision(
    collision,
    traces_with_margin,
    start_frame,
    end_frame,
    left,
    right,
    nbefore,
    all_templates,
    sparsity_mask,
    cut_out_before,
    cut_out_after,
):
    """
    Compute the best fit for a collision between a spike and its overlapping spikes.
    The function first cuts out the traces around the spike and its overlapping spikes, then
    fits a multi-linear regression model to the traces using the centered templates as predictors.

    Parameters
    ----------
    collision: np.ndarray
        A numpy array of shape (n_colliding_spikes, ) containing the colliding spikes (spike_dtype).
    traces_with_margin: np.ndarray
        A numpy array of shape (n_samples, n_channels) containing the traces with a margin.
    start_frame: int
        The start frame of the chunk for traces_with_margin.
    end_frame: int
        The end frame of the chunk for traces_with_margin.
    left: int
        The left margin of the chunk for traces_with_margin.
    right: int
        The right margin of the chunk for traces_with_margin.
    nbefore: int
        The number of samples before the spike to consider for the fit.
    all_templates: np.ndarray
        A numpy array of shape (n_units, n_samples, n_channels) containing the templates.
    sparsity_mask: boolean mask
        The sparsity mask
    cut_out_before: int
        The number of samples to cut out before the spike.
    cut_out_after: int
        The number of samples to cut out after the spike.

    Returns
    -------
    np.ndarray
        The fitted scaling factors for the colliding spikes.
    """
    from sklearn.linear_model import LinearRegression

    # make center of the spike externally
    sample_first_centered = np.min(collision["sample_index"]) - (start_frame - left)
    sample_last_centered = np.max(collision["sample_index"]) - (start_frame - left)

    # construct sparsity as union between units' sparsity
    common_sparse_mask = np.zeros(sparsity_mask.shape[1], dtype="int")
    for spike in collision:
        mask_i = sparsity_mask[spike["unit_index"]]
        common_sparse_mask = np.logical_or(common_sparse_mask, mask_i)
    (sparse_indices,) = np.nonzero(common_sparse_mask)

    local_waveform_start = max(0, sample_first_centered - cut_out_before)
    local_waveform_end = min(traces_with_margin.shape[0], sample_last_centered + cut_out_after)
    local_waveform = traces_with_margin[local_waveform_start:local_waveform_end, sparse_indices]
    num_samples_local_waveform = local_waveform.shape[0]

    y = local_waveform.T.flatten()
    X = np.zeros((len(y), len(collision)))
    for i, spike in enumerate(collision):
        full_template = np.zeros_like(local_waveform)
        # center wrt cutout traces
        sample_centered = spike["sample_index"] - (start_frame - left) - local_waveform_start
        template = all_templates[spike["unit_index"]][:, sparse_indices]
        template_cut = template[nbefore - cut_out_before : nbefore + cut_out_after]
        # deal with borders
        if sample_centered - cut_out_before < 0:
            full_template[: sample_centered + cut_out_after] = template_cut[cut_out_before - sample_centered :]
        elif sample_centered + cut_out_after > num_samples_local_waveform:
            full_template[sample_centered - cut_out_before :] = template_cut[
                : -(cut_out_after + sample_centered - num_samples_local_waveform)
            ]
        else:
            full_template[sample_centered - cut_out_before : sample_centered + cut_out_after] = template_cut
        X[:, i] = full_template.T.flatten()

    reg = LinearRegression(fit_intercept=True, positive=True).fit(X, y)
    scalings = reg.coef_
    return scalings


# uncomment for debugging
# def plot_collisions(we, sparsity=None, num_collisions=None):
#     """
#     Plot the fitting of collision spikes.

#     Parameters
#     ----------
#     we : WaveformExtractor
#         The WaveformExtractor object.
#     sparsity : ChannelSparsity, default: None
#         The ChannelSparsity. If None, only main channels are plotted.
#     num_collisions : int, default: None
#         Number of collisions to plot. If None, all collisions are plotted.
#     """
#     assert we.is_extension("amplitude_scalings"), "Could not find amplitude scalings extension!"
#     sac = we.load_extension("amplitude_scalings")
#     handle_collisions = sac._params["handle_collisions"]
#     assert handle_collisions, "Amplitude scalings was run without handling collisions!"
#     scalings = sac.get_data()

#     # overlapping_mask = sac.overlapping_mask
#     # num_collisions = num_collisions or len(overlapping_mask)
#     spikes = sac.spikes
#     collisions = sac._extension_data[f"collisions"]
#     collision_keys = list(collisions.keys())
#     num_collisions = num_collisions or len(collisions)
#     num_collisions = min(num_collisions, len(collisions))

#     for i in range(num_collisions):
#         overlapping_spikes = collisions[collision_keys[i]]
#         ax = plot_one_collision(
#             we, collision_keys[i], overlapping_spikes, spikes, scalings=scalings, sparsity=sparsity
#         )


# def plot_one_collision(
#     we,
#     spike_index,
#     overlapping_spikes,
#     spikes,
#     scalings=None,
#     sparsity=None,
#     cut_out_samples=100,
#     ax=None
# ):
#     import matplotlib.pyplot as plt

#     if ax is None:
#         fig, ax = plt.subplots()

#     recording = we.recording
#     nbefore_nafter_max = max(we.nafter, we.nbefore)
#     cut_out_samples = max(cut_out_samples, nbefore_nafter_max)

#     if sparsity is not None:
#         unit_inds_to_channel_indices = sparsity.unit_id_to_channel_indices
#         sparse_indices = np.array([], dtype="int")
#         for spike in overlapping_spikes:
#             sparse_indices_i = unit_inds_to_channel_indices[we.unit_ids[spike["unit_index"]]]
#             sparse_indices = np.union1d(sparse_indices, sparse_indices_i)
#     else:
#         sparse_indices = np.unique(overlapping_spikes["channel_index"])

#     channel_ids = recording.channel_ids[sparse_indices]

#     center_spike = overlapping_spikes[0]
#     max_delta = np.max(
#         [
#             np.abs(center_spike["sample_index"] - np.min(overlapping_spikes[1:]["sample_index"])),
#             np.abs(center_spike["sample_index"] - np.max(overlapping_spikes[1:]["sample_index"])),
#         ]
#     )
#     sf = max(0, center_spike["sample_index"] - max_delta - cut_out_samples)
#     ef = min(
#         center_spike["sample_index"] + max_delta + cut_out_samples,
#         recording.get_num_samples(segment_index=center_spike["segment_index"]),
#     )
#     tr_overlap = recording.get_traces(start_frame=sf, end_frame=ef, channel_ids=channel_ids, return_scaled=True)
#     ts = np.arange(sf, ef) / recording.sampling_frequency * 1000
#     max_tr = np.max(np.abs(tr_overlap))

#     for ch, tr in enumerate(tr_overlap.T):
#         _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="k")
#         ax.text(ts[0], 1.2 * ch * max_tr - 0.3 * max_tr, f"Ch:{channel_ids[ch]}")

#     used_labels = []
#     for i, spike in enumerate(overlapping_spikes):
#         label = f"U{spike['unit_index']}"
#         if label in used_labels:
#             label = None
#         else:
#             used_labels.append(label)
#         ax.axvline(
#             spike["sample_index"] / recording.sampling_frequency * 1000, color=f"C{spike['unit_index']}", label=label
#         )

#     if scalings is not None:
#         fitted_traces = np.zeros_like(tr_overlap)

#         all_templates = we.get_all_templates()
#         for i, spike in enumerate(overlapping_spikes):
#             template = all_templates[spike["unit_index"]]
#             overlap_index = np.where(spikes == spike)[0][0]
#             template_scaled = scalings[overlap_index] * template
#             template_scaled_sparse = template_scaled[:, sparse_indices]
#             sample_start = spike["sample_index"] - we.nbefore
#             sample_end = sample_start + template_scaled_sparse.shape[0]

#             fitted_traces[sample_start - sf : sample_end - sf] += template_scaled_sparse

#             for ch, temp in enumerate(template_scaled_sparse.T):
#                 ts_template = np.arange(sample_start, sample_end) / recording.sampling_frequency * 1000
#                 _ = ax.plot(ts_template, temp + 1.2 * ch * max_tr, color=f"C{spike['unit_index']}", ls="--")

#         for ch, tr in enumerate(fitted_traces.T):
#             _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="gray", alpha=0.7)

#         fitted_line = ax.get_lines()[-1]
#         fitted_line.set_label("Fitted")

#     ax.legend()
#     ax.set_title(f"Spike {spike_index} - sample {center_spike['sample_index']}")
#     return ax
