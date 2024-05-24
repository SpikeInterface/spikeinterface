from __future__ import annotations

import numpy as np

from spikeinterface.core import ChannelSparsity, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension

from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, run_node_pipeline, find_parent_of_type

from ..core.template_tools import get_dense_templates_array, _get_nbefore

# DEBUG = True


# TODO extra sparsity and job_kwargs handling


class ComputeAmplitudeScalings(AnalyzerExtension):
    """
    Computes the amplitude scalings from a SortingAnalyzer.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    sparsity: ChannelSparsity or None, default: None
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
    max_dense_channels: int, default: 16
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
    ms_before : float or None, default: None
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the SortingAnalyzer ms_before is used.
    ms_after : float or None, default: None
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the SortingAnalyzer ms_after is used.
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

    extension_name = "amplitude_scalings"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = True
    nodepipeline_variables = ["amplitude_scalings", "collision_mask"]
    need_job_kwargs = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

        self.collisions = None

    def _set_params(
        self,
        sparsity=None,
        max_dense_channels=16,
        ms_before=None,
        ms_after=None,
        handle_collisions=True,
        delta_collision_ms=2,
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
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["amplitude_scalings"] = self.data["amplitude_scalings"][keep_spike_mask]
        if self.params["handle_collisions"]:
            new_data["collision_mask"] = self.data["collision_mask"][keep_spike_mask]
        return new_data

    def _get_pipeline_nodes(self):

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting

        return_scaled = self.sorting_analyzer.return_scaled

        all_templates = get_dense_templates_array(self.sorting_analyzer, return_scaled=return_scaled)
        nbefore = _get_nbefore(self.sorting_analyzer)
        nafter = all_templates.shape[1] - nbefore

        # if ms_before / ms_after are set in params then the original templates are shorten
        if self.params["ms_before"] is not None:
            cut_out_before = int(self.params["ms_before"] * self.sorting_analyzer.sampling_frequency / 1000.0)
            assert (
                cut_out_before <= nbefore
            ), f"`ms_before` must be smaller than `ms_before` used in ComputeTemplates: {nbefore}"
        else:
            cut_out_before = nbefore

        if self.params["ms_after"] is not None:
            cut_out_after = int(self.params["ms_after"] * self.sorting_analyzer.sampling_frequency / 1000.0)
            assert (
                cut_out_after <= nafter
            ), f"`ms_after` must be smaller than `ms_after` used in WaveformExractor: {we._params['ms_after']}"
        else:
            cut_out_after = nafter

        peak_sign = "neg" if np.abs(np.min(all_templates)) > np.max(all_templates) else "pos"
        extremum_channels_indices = get_template_extremum_channel(
            self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        )

        # collisions
        handle_collisions = self.params["handle_collisions"]
        delta_collision_ms = self.params["delta_collision_ms"]
        delta_collision_samples = int(delta_collision_ms / 1000 * self.sorting_analyzer.sampling_frequency)

        if self.sorting_analyzer.is_sparse() and self.params["sparsity"] is None:
            sparsity = self.sorting_analyzer.sparsity
        elif self.sorting_analyzer.is_sparse() and self.params["sparsity"] is not None:
            sparsity = self.params["sparsity"]
            # assert provided sparsity is sparser than the one in the waveform extractor
            waveform_sparsity = self.sorting_analyzer.sparsity
            assert np.all(
                np.sum(waveform_sparsity.mask, 1) - np.sum(sparsity.mask, 1) > 0
            ), "The provided sparsity needs to be sparser than the one in the waveform extractor!"
        elif not self.sorting_analyzer.is_sparse() and self.params["sparsity"] is not None:
            sparsity = self.params["sparsity"]
        else:
            if self.params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self.params["max_dense_channels"], ""
            sparsity = ChannelSparsity.create_dense(self.sorting_analyzer)
        sparsity_mask = sparsity.mask

        spike_retriever_node = SpikeRetriever(
            recording,
            sorting,
            channel_from_template=True,
            extremum_channel_inds=extremum_channels_indices,
            include_spikes_in_margin=True,
        )
        amplitude_scalings_node = AmplitudeScalingNode(
            recording,
            parents=[spike_retriever_node],
            return_output=True,
            all_templates=all_templates,
            sparsity_mask=sparsity_mask,
            nbefore=nbefore,
            nafter=nafter,
            cut_out_before=cut_out_before,
            cut_out_after=cut_out_after,
            return_scaled=return_scaled,
            handle_collisions=handle_collisions,
            delta_collision_samples=delta_collision_samples,
        )
        nodes = [spike_retriever_node, amplitude_scalings_node]
        return nodes

    def _run(self, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        amp_scalings, collision_mask = run_node_pipeline(
            self.sorting_analyzer.recording,
            nodes,
            job_kwargs=job_kwargs,
            job_name="amplitude_scalings",
            gather_mode="memory",
        )
        self.data["amplitude_scalings"] = amp_scalings
        if self.params["handle_collisions"]:
            self.data["collision_mask"] = collision_mask
            # TODO: make collisions "global"
            # for collision in collisions:
            #     collisions_dict.update(collision)
            # self.collisions = collisions_dict
            # # Note: collisions are note in _extension_data because they are not pickable. We only store the indices
            # self._extension_data["collisions"] = np.array(list(collisions_dict.keys()))

    def _get_data(self):
        return self.data[f"amplitude_scalings"]


register_result_extension(ComputeAmplitudeScalings)
compute_amplitude_scalings = ComputeAmplitudeScalings.function_factory()


class AmplitudeScalingNode(PipelineNode):
    def __init__(
        self,
        recording,
        parents,
        return_output,
        all_templates,
        sparsity_mask,
        nbefore,
        nafter,
        cut_out_before,
        cut_out_after,
        return_scaled,
        handle_collisions,
        delta_collision_samples,
    ):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.return_scaled = return_scaled
        if return_scaled and recording.has_scaled():
            self._dtype = np.float32
            self._gains = recording.get_channel_gains()
            self._offsets = recording.get_channel_gains()
        else:
            self._dtype = recording.get_dtype()
            self._gains = None
            self._offsets = None
        spike_retriever = find_parent_of_type(parents, SpikeRetriever)
        assert isinstance(
            spike_retriever, SpikeRetriever
        ), "SpikeAmplitudeNode needs a single SpikeRetriever as a parent"
        assert spike_retriever.include_spikes_in_margin, "Need SpikeRetriever with include_spikes_in_margin=True"
        if not handle_collisions:
            self._margin = max(nbefore, nafter)
        else:
            # in this case we extend the margin to be able to get with collisions outside the chunk
            margin_waveforms = max(nbefore, nafter)
            max_margin_collisions = delta_collision_samples + margin_waveforms
            self._margin = max_margin_collisions

        self._all_templates = all_templates
        self._sparsity_mask = sparsity_mask
        self._nbefore = nbefore
        self._nafter = nafter
        self._cut_out_before = cut_out_before
        self._cut_out_after = cut_out_after
        self._handle_collisions = handle_collisions
        self._delta_collision_samples = delta_collision_samples

        self._kwargs.update(
            all_templates=all_templates,
            sparsity_mask=sparsity_mask,
            nbefore=nbefore,
            nafter=nafter,
            cut_out_before=cut_out_before,
            cut_out_after=cut_out_after,
            return_scaled=return_scaled,
            handle_collisions=handle_collisions,
            delta_collision_samples=delta_collision_samples,
        )

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        from scipy.stats import linregress

        # scale traces with margin to match scaling of templates
        if self._gains is not None:
            traces = traces.astype("float32") * self._gains + self._offsets

        all_templates = self._all_templates
        sparsity_mask = self._sparsity_mask
        nbefore = self._nbefore
        cut_out_before = self._cut_out_before
        cut_out_after = self._cut_out_after
        handle_collisions = self._handle_collisions
        delta_collision_samples = self._delta_collision_samples

        # local_spikes_w_margin = peaks
        # i0 = np.searchsorted(local_spikes_w_margin["sample_index"], left_margin)
        # i1 = np.searchsorted(local_spikes_w_margin["sample_index"], traces.shape[0] - right_margin)
        # local_spikes = local_spikes_w_margin[i0:i1]

        local_spikes_w_margin = peaks
        local_spikes = local_spikes_w_margin[~peaks["in_margin"]]

        # set colliding spikes apart (if needed)
        if handle_collisions:
            # local spikes with margin!
            collisions = find_collisions(local_spikes, local_spikes_w_margin, delta_collision_samples, sparsity_mask)
        else:
            collisions = {}

        # compute the scaling for each spike
        scalings = np.zeros(len(local_spikes), dtype=float)
        spike_collision_mask = np.zeros(len(local_spikes), dtype=bool)

        for spike_index, spike in enumerate(local_spikes):
            if spike_index in collisions.keys():
                # we deal with overlapping spikes later
                continue
            unit_index = spike["unit_index"]
            sample_centered = spike["sample_index"]
            (sparse_indices,) = np.nonzero(sparsity_mask[unit_index])
            template = all_templates[unit_index][:, sparse_indices]
            template = template[nbefore - cut_out_before : nbefore + cut_out_after]
            cut_out_start = sample_centered - cut_out_before
            cut_out_end = sample_centered + cut_out_after
            if sample_centered - cut_out_before < 0:
                local_waveform = traces[:cut_out_end, sparse_indices]
                template = template[cut_out_before - sample_centered :]
            elif sample_centered + cut_out_after > traces.shape[0]:
                local_waveform = traces[cut_out_start:, sparse_indices]
                template = template[: -(sample_centered + cut_out_after - (traces.shape[0]))]
            else:
                local_waveform = traces[cut_out_start:cut_out_end, sparse_indices]
            assert template.shape == local_waveform.shape

            # here we use linregress, which is equivalent to using sklearn LinearRegression with fit_intercept=True
            # y = local_waveform.flatten()
            # X = template.flatten()[:, np.newaxis]
            # reg = LinearRegression(positive=True, fit_intercept=True).fit(X, y)
            # scalings[spike_index] = reg.coef_[0]

            # closed form: W = (X' * X)^-1 X' y
            # y = local_waveform.flatten()[:, None]
            # X = np.ones((len(y), 2))
            # X[:, 0] = template.flatten()
            # W = np.linalg.inv(X.T @ X) @ X.T @ y
            # scalings[spike_index] = W[0, 0]

            linregress_res = linregress(template.flatten(), local_waveform.flatten())
            scalings[spike_index] = linregress_res[0]

        # deal with collisions
        if len(collisions) > 0:
            for spike_index, collision in collisions.items():
                scaled_amps = fit_collision(
                    collision,
                    traces,
                    nbefore,
                    all_templates,
                    sparsity_mask,
                    cut_out_before,
                    cut_out_after,
                )
                # the scaling for the current spike is at index 0
                scalings[spike_index] = scaled_amps[0]
                spike_collision_mask[spike_index] = True

        # TODO: switch to collision mask and return that (to use concatenation)
        return (scalings, spike_collision_mask)

    def get_trace_margin(self):
        return self._margin


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
    sample_first_centered = np.min(collision["sample_index"])
    sample_last_centered = np.max(collision["sample_index"])

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
        sample_centered = spike["sample_index"] - local_waveform_start
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
#     we : SortingAnalyzer
#         The SortingAnalyzer object.
#     sparsity : ChannelSparsity, default=None
#         The ChannelSparsity. If None, only main channels are plotted.
#     num_collisions : int, default=None
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
