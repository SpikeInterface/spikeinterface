import numpy as np

from spikeinterface.core import ChannelSparsity, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


class AmplitudeScalingsCalculator(BaseWaveformExtractorExtension):
    """
    Computes amplitude scalings from WaveformExtractor.
    """

    extension_name = "amplitude_scalings"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(
            extremum_channel_inds=extremum_channel_inds, use_cache=False
        )

    def _set_params(
        self,
        sparsity,
        max_dense_channels,
        ms_before,
        ms_after,
        handle_collisions,
        max_consecutive_collisions,
        delta_collision_ms,
    ):
        params = dict(
            sparsity=sparsity,
            max_dense_channels=max_dense_channels,
            ms_before=ms_before,
            ms_after=ms_after,
            handle_collisions=handle_collisions,
            max_consecutive_collisions=max_consecutive_collisions,
            delta_collision_ms=delta_collision_ms,
        )
        return params

    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.in1d(old_unit_ids, unit_ids))

        spike_mask = np.in1d(self.spikes["unit_index"], unit_inds)
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
        max_consecutive_collisions = self._params["max_consecutive_collisions"]
        delta_collision_ms = self._params["delta_collision_ms"]
        delta_collision_samples = int(delta_collision_ms / 1000 * we.sampling_frequency)

        return_scaled = we._params["return_scaled"]
        unit_ids = we.unit_ids

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

        if we.is_sparse():
            sparsity = we.sparsity
        elif self._params["sparsity"] is not None:
            sparsity = self._params["sparsity"]
        else:
            if self._params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self._params["max_dense_channels"], ""
            sparsity = ChannelSparsity.create_dense(we)
        sparsity_inds = sparsity.unit_id_to_channel_indices

        # easier to use in chunk function as spikes use unit_index instead o id
        unit_inds_to_channel_indices = {unit_ind: sparsity_inds[unit_id] for unit_ind, unit_id in enumerate(unit_ids)}
        all_templates = we.get_all_templates()

        # precompute segment slice
        segment_slices = []
        for segment_index in range(we.get_num_segments()):
            i0 = np.searchsorted(self.spikes["segment_index"], segment_index)
            i1 = np.searchsorted(self.spikes["segment_index"], segment_index + 1)
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
            unit_inds_to_channel_indices,
            nbefore,
            nafter,
            cut_out_before,
            cut_out_after,
            return_scaled,
            handle_collisions,
            max_consecutive_collisions,
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
        (amp_scalings,) = zip(*out)
        amp_scalings = np.concatenate(amp_scalings)

        self._extension_data[f"amplitude_scalings"] = amp_scalings

    def get_data(self, outputs="concatenated"):
        """
        Get computed spike amplitudes.
        Parameters
        ----------
        outputs : str, optional
            'concatenated' or 'by_unit', by default 'concatenated'
        Returns
        -------
        spike_amplitudes : np.array or dict
            The spike amplitudes as an array (outputs='concatenated') or
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
    handle_collisions=False,
    max_consecutive_collisions=3,
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
    sparsity: ChannelSparsity, default: None
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
    max_dense_channels: int, default: 16
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
    ms_before : float, default: None
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_before is used.
    ms_after : float, default: None
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_after is used.
    handle_collisions: bool, default: False
        Whether to handle collisions between spikes. If True, the amplitude scaling of colliding spikes
        (defined as spikes within `delta_collision_ms` ms and with overlapping sparsity) is computed by fitting a
        multi-linear regression model (with `sklearn.LinearRegression`). If False, each spike is fitted independently.
    max_consecutive_collisions: int, default: 3
        The maximum number of consecutive collisions to handle on each side of a spike.
    delta_collision_ms: float, default: 2
        The maximum time difference in ms between two spikes to be considered as colliding.
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    outputs: str, default: 'concatenated'
        How the output should be returned:
            - 'concatenated'
            - 'by_unit'
    {}

    Returns
    -------
    amplitude_scalings: np.array or list of dict
        The amplitude scalings.
            - If 'concatenated' all amplitudes for all spikes and all units are concatenated
            - If 'by_unit', amplitudes are returned as a list (for segments) of dictionaries (for units)
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
            max_consecutive_collisions=max_consecutive_collisions,
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
    unit_inds_to_channel_indices,
    nbefore,
    nafter,
    cut_out_before,
    cut_out_after,
    return_scaled,
    handle_collisions,
    max_consecutive_collisions,
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
    worker_ctx["unit_inds_to_channel_indices"] = unit_inds_to_channel_indices
    worker_ctx["handle_collisions"] = handle_collisions
    worker_ctx["max_consecutive_collisions"] = max_consecutive_collisions
    worker_ctx["delta_collision_samples"] = delta_collision_samples

    if not handle_collisions:
        worker_ctx["margin"] = max(nbefore, nafter)
    else:
        margin_waveforms = max(nbefore, nafter)
        max_margin_collisions = int(max_consecutive_collisions * delta_collision_samples)
        worker_ctx["margin"] = max(margin_waveforms, max_margin_collisions)

    return worker_ctx


def _amplitude_scalings_chunk(segment_index, start_frame, end_frame, worker_ctx):
    from scipy.stats import linregress

    # recover variables of the worker
    spikes = worker_ctx["spikes"]
    recording = worker_ctx["recording"]
    all_templates = worker_ctx["all_templates"]
    segment_slices = worker_ctx["segment_slices"]
    unit_inds_to_channel_indices = worker_ctx["unit_inds_to_channel_indices"]
    nbefore = worker_ctx["nbefore"]
    cut_out_before = worker_ctx["cut_out_before"]
    cut_out_after = worker_ctx["cut_out_after"]
    margin = worker_ctx["margin"]
    return_scaled = worker_ctx["return_scaled"]
    handle_collisions = worker_ctx["handle_collisions"]
    max_consecutive_collisions = worker_ctx["max_consecutive_collisions"]
    delta_collision_samples = worker_ctx["delta_collision_samples"]

    spikes_in_segment = spikes[segment_slices[segment_index]]

    i0 = np.searchsorted(spikes_in_segment["sample_index"], start_frame)
    i1 = np.searchsorted(spikes_in_segment["sample_index"], end_frame)

    local_waveforms = []
    templates = []
    scalings = []

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
            overlapping_mask = _find_overlapping_mask(
                local_spikes, max_consecutive_collisions, delta_collision_samples, unit_inds_to_channel_indices
            )
            overlapping_spike_indices = overlapping_mask[:, max_consecutive_collisions]
            print(
                f"Found {len(overlapping_spike_indices)} overlapping spikes in segment {segment_index}! - chunk {start_frame} - {end_frame}"
            )
        else:
            overlapping_spike_indices = np.array([], dtype=int)

        # get all waveforms
        scalings = np.zeros(len(local_spikes), dtype=float)
        for spike_index, spike in enumerate(local_spikes):
            if spike_index in overlapping_spike_indices:
                # we deal with overlapping spikes later
                continue
            unit_index = spike["unit_index"]
            sample_index = spike["sample_index"]
            sparse_indices = unit_inds_to_channel_indices[unit_index]
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
                template = template[: -(sample_index + cut_out_after - end_frame)]
            else:
                local_waveform = traces_with_margin[cut_out_start:cut_out_end, sparse_indices]
            assert template.shape == local_waveform.shape
            local_waveforms.append(local_waveform)
            templates.append(template)
            linregress_res = linregress(template.flatten(), local_waveform.flatten())
            scalings[spike_index] = linregress_res[0]

        # deal with collisions
        if len(overlapping_spike_indices) > 0:
            for overlapping in overlapping_mask:
                spike_index = overlapping[max_consecutive_collisions]
                overlapping_spikes = local_spikes[overlapping[overlapping >= 0]]
                scaled_amps = _fit_collision(
                    overlapping_spikes,
                    traces_with_margin,
                    start_frame,
                    end_frame,
                    left,
                    right,
                    nbefore,
                    all_templates,
                    unit_inds_to_channel_indices,
                    cut_out_before,
                    cut_out_after,
                )
                # get the right amplitude scaling
                scalings[spike_index] = scaled_amps[np.where(overlapping >= 0)[0] == max_consecutive_collisions]

    return (scalings,)


### Collision handling ###
def _are_unit_indices_overlapping(unit_inds_to_channel_indices, i, j):
    """
    Returns True if the unit indices i and j are overlapping, False otherwise

    Parameters
    ----------
    unit_inds_to_channel_indices: dict
        A dictionary mapping unit indices to channel indices
    i: int
        The first unit index
    j: int
        The second unit index

    Returns
    -------
    bool
        True if the unit indices i and j are overlapping, False otherwise
    """
    if len(np.intersect1d(unit_inds_to_channel_indices[i], unit_inds_to_channel_indices[j])) > 0:
        return True
    else:
        return False


def _find_overlapping_mask(spikes, max_consecutive_spikes, delta_overlap_samples, unit_inds_to_channel_indices):
    """
    Finds the overlapping spikes for each spike in spikes and returns a boolean mask of shape
    (n_spikes, 2 * max_consecutive_spikes + 1).

    Parameters
    ----------
    spikes: np.array
        An array of spikes
    max_consecutive_spikes: int
        The maximum number of consecutive spikes to consider
    delta_overlap_samples: int
        The maximum number of samples between two spikes to consider them as overlapping
    unit_inds_to_channel_indices: dict
        A dictionary mapping unit indices to channel indices

    Returns
    -------
    overlapping_mask: np.array
        A boolean mask of shape (n_spikes, 2 * max_consecutive_spikes + 1) where the central column (max_consecutive_spikes)
        is the current spike index, while the other columns are the indices of the overlapping spikes. The first
        max_consecutive_spikes columns are the pre-overlapping spikes, while the last max_consecutive_spikes columns are
        the post-overlapping spikes.
    """

    # overlapping_mask is a matrix of shape (n_spikes, 2 * max_consecutive_spikes + 1)
    # the central column (max_consecutive_spikes) is the current spike index, while the other columns are the
    # indices of the overlapping spikes. The first max_consecutive_spikes columns are the pre-overlapping spikes,
    # while the last max_consecutive_spikes columns are the post-overlapping spikes
    # Rows with all -1 are non-colliding spikes and are removed later
    overlapping_mask_full = -1 * np.ones((len(spikes), 2 * max_consecutive_spikes + 1), dtype=int)
    overlapping_mask_full[:, max_consecutive_spikes] = np.arange(len(spikes))

    for i, spike in enumerate(spikes):
        # find the possible spikes per and post within max_consecutive_spikes * delta_overlap_samples
        consecutive_window_pre = np.searchsorted(
            spikes["sample_index"],
            spike["sample_index"] - max_consecutive_spikes * delta_overlap_samples,
        )
        consecutive_window_post = np.searchsorted(
            spikes["sample_index"],
            spike["sample_index"] + max_consecutive_spikes * delta_overlap_samples,
        )
        pre_possible_consecutive_spikes = spikes[consecutive_window_pre:i][::-1]
        post_possible_consecutive_spikes = spikes[i + 1 : consecutive_window_post]

        # here we fill in the overlapping information by consecutively looping through the possible consecutive spikes
        # and checking the spatial overlap and the delay with the previous overlapping spike
        # pre and post are hanlded separately. Note that the pre-spikes are already sorted backwards

        # overlap_rank keeps track of the rank of consecutive collisions (i.e., rank 0 is the first, rank 1 is the second, etc.)
        # this is needed because we are just considering spikes with spatial overlap, while the possible consecutive spikes
        # only looked at the temporal overlap
        overlap_rank = 0
        if len(pre_possible_consecutive_spikes) > 0:
            for c_pre, spike_consecutive_pre in enumerate(pre_possible_consecutive_spikes[::-1]):
                if _are_unit_indices_overlapping(
                    unit_inds_to_channel_indices, spike["unit_index"], spike_consecutive_pre["unit_index"]
                ):
                    if (
                        spikes[overlapping_mask_full[i, max_consecutive_spikes - overlap_rank]]["sample_index"]
                        - spike_consecutive_pre["sample_index"]
                        < delta_overlap_samples
                    ):
                        overlapping_mask_full[i, max_consecutive_spikes - overlap_rank - 1] = i - 1 - c_pre
                        overlap_rank += 1
                    else:
                        break
        # if overlap_rank > 1:
        #     print(f"\tHigher order pre-overlap for spike {i}!")

        overlap_rank = 0
        if len(post_possible_consecutive_spikes) > 0:
            for c_post, spike_consecutive_post in enumerate(post_possible_consecutive_spikes):
                if _are_unit_indices_overlapping(
                    unit_inds_to_channel_indices, spike["unit_index"], spike_consecutive_post["unit_index"]
                ):
                    if (
                        spike_consecutive_post["sample_index"]
                        - spikes[overlapping_mask_full[i, max_consecutive_spikes + overlap_rank]]["sample_index"]
                        < delta_overlap_samples
                    ):
                        overlapping_mask_full[i, max_consecutive_spikes + overlap_rank + 1] = i + 1 + c_post
                        overlap_rank += 1
                    else:
                        break
        # if overlap_rank > 1:
        #     print(f"\tHigher order post-overlap for spike {i}!")

        # in case no collisions were found, we set the central column to -1 so that we can easily identify the non-colliding spikes
        if np.sum(overlapping_mask_full[i] != -1) == 1:
            overlapping_mask_full[i, max_consecutive_spikes] = -1

        # only return rows with collisions
        overlapping_inds = []
        for i, overlapping in enumerate(overlapping_mask_full):
            if np.any(overlapping >= 0):
                overlapping_inds.append(i)
        overlapping_mask = overlapping_mask_full[overlapping_inds]

    return overlapping_mask


def _fit_collision(
    overlapping_spikes,
    traces_with_margin,
    start_frame,
    end_frame,
    left,
    right,
    nbefore,
    all_templates,
    unit_inds_to_channel_indices,
    cut_out_before,
    cut_out_after,
):
    """ """
    from sklearn.linear_model import LinearRegression

    sample_first_centered = overlapping_spikes[0]["sample_index"] - start_frame - left
    sample_last_centered = overlapping_spikes[-1]["sample_index"] - start_frame - left

    # construct sparsity as union between units' sparsity
    sparse_indices = np.array([], dtype="int")
    for spike in overlapping_spikes:
        sparse_indices_i = unit_inds_to_channel_indices[spike["unit_index"]]
        sparse_indices = np.union1d(sparse_indices, sparse_indices_i)

    local_waveform_start = max(0, sample_first_centered - cut_out_before)
    local_waveform_end = min(traces_with_margin.shape[0], sample_last_centered + cut_out_after)
    local_waveform = traces_with_margin[local_waveform_start:local_waveform_end, sparse_indices]

    y = local_waveform.T.flatten()
    X = np.zeros((len(y), len(overlapping_spikes)))
    for i, spike in enumerate(overlapping_spikes):
        full_template = np.zeros_like(local_waveform)
        # center wrt cutout traces
        sample_centered = spike["sample_index"] - local_waveform_start
        template = all_templates[spike["unit_index"]][:, sparse_indices]
        template_cut = template[nbefore - cut_out_before : nbefore + cut_out_after]
        # deal with borders
        if sample_centered - cut_out_before < 0:
            full_template[: sample_centered + cut_out_after] = template_cut[cut_out_before - sample_centered :]
        elif sample_centered + cut_out_after > end_frame + right:
            full_template[sample_centered - cut_out_before :] = template_cut[: -cut_out_after - (end_frame + right)]
        else:
            full_template[sample_centered - cut_out_before : sample_centered + cut_out_after] = template_cut
        X[:, i] = full_template.T.flatten()

    reg = LinearRegression().fit(X, y)
    amps = reg.coef_
    return amps


# TODO: fix this!
# def plot_overlapping_spikes(we, overlap,
#                             spikes, cut_out_samples=100,
#                             max_consecutive_spikes=3,
#                             sparsity=None,
#                             fitted_amps=None):
#     recording = we.recording
#     nbefore_nafter_max = max(we.nafter, we.nbefore)
#     cut_out_samples = max(cut_out_samples, nbefore_nafter_max)
#     spike_index = overlap[max_consecutive_spikes]
#     overlap_indices = overlap[overlap != -1]
#     overlapping_spikes = spikes[overlap_indices]

#     if sparsity is not None:
#         unit_inds_to_channel_indices = sparsity.unit_id_to_channel_indices
#         sparse_indices = np.array([], dtype="int")
#         for spike in overlapping_spikes:
#             sparse_indices_i = unit_inds_to_channel_indices[we.unit_ids[spike["unit_index"]]]
#             sparse_indices = np.union1d(sparse_indices, sparse_indices_i)
#     else:
#         sparse_indices = np.unique(overlapping_spikes["channel_index"])

#     channel_ids = recording.channel_ids[sparse_indices]

#     center_spike = spikes[spike_index]["sample_index"]
#     max_delta = np.max([np.abs(center_spike - overlapping_spikes[0]["sample_index"]),
#                         np.abs(center_spike - overlapping_spikes[-1]["sample_index"])])
#     sf = center_spike - max_delta - cut_out_samples
#     ef = center_spike + max_delta + cut_out_samples
#     tr_overlap = recording.get_traces(start_frame=sf,
#                                       end_frame=ef,
#                                       channel_ids=channel_ids, return_scaled=True)
#     ts = np.arange(sf, ef) / recording.sampling_frequency * 1000
#     max_tr = np.max(np.abs(tr_overlap))
#     fig, ax = plt.subplots()
#     for ch, tr in enumerate(tr_overlap.T):
#         _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="k")
#         ax.text(ts[0], 1.2 * ch * max_tr - 0.3 * max_tr, f"Ch:{channel_ids[ch]}")

#     used_labels = []
#     for spike in overlapping_spikes:
#         label = f"U{spike['unit_index']}"
#         if label in used_labels:
#             label = None
#         else:
#             used_labels.append(label)
#         ax.axvline(spike["sample_index"] / recording.sampling_frequency * 1000,
#                    color=f"C{spike['unit_index']}", label=label)

#     if fitted_amps is not None:
#         fitted_traces = np.zeros_like(tr_overlap)

#         all_templates = we.get_all_templates()
#         for i, spike in enumerate(overlapping_spikes):
#             template = all_templates[spike["unit_index"]]
#             template_scaled = fitted_amps[overlap_indices[i]] * template
#             template_scaled_sparse = template_scaled[:, sparse_indices]
#             sample_start = spike["sample_index"] - we.nbefore
#             sample_end = sample_start + template_scaled_sparse.shape[0]

#             fitted_traces[sample_start - sf: sample_end - sf] += template_scaled_sparse

#             for ch, temp in enumerate(template_scaled_sparse.T):

#                 ts_template = np.arange(sample_start, sample_end) / recording.sampling_frequency * 1000
#                 _ = ax.plot(ts_template, temp + 1.2 * ch * max_tr, color=f"C{spike['unit_index']}",
#                             ls="--")

#         for ch, tr in enumerate(fitted_traces.T):
#             _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="gray", alpha=0.7)

#         fitted_line = ax.get_lines()[-1]
#         fitted_line.set_label("Fitted")


#     ax.legend()
#     ax.set_title(f"Spike {spike_index} - sample {center_spike}")
#     return tr_overlap, ax
