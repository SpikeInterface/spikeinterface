import numpy as np
from scipy.stats import linregress

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
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)

    def _set_params(self, sparsity, max_dense_channels, ms_before, ms_after,
                    handle_collisions, collision_ms):
        params = dict(sparsity=sparsity, max_dense_channels=max_dense_channels,
                      ms_before=ms_before, ms_after=ms_after, handle_collisions=handle_collisions,
                      collision_ms=collision_ms)
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
        handle_collisions = self._params["handle_collisions"]

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
        unit_inds_to_channel_indices = {
            unit_ind: sparsity_inds[unit_id] for unit_ind, unit_id in enumerate(unit_ids)
        }
        all_templates = we.get_all_templates()

        # precompute segment slice
        segment_slices = []
        for segment_index in range(we.get_num_segments()):
            i0 = np.searchsorted(self.spikes['segment_index'], segment_index)
            i1 = np.searchsorted(self.spikes['segment_index'], segment_index + 1)
            segment_slices.append(slice(i0, i1))

        # handle collisions
        if handle_collisions:
            import networkx as nx

            collision_ms = self._params["collision_ms"]
            delta_samples = int(collision_ms * we.sampling_frequency / 1000)

            # find overlapping "cliques": cliques are connected cycles in a graph
            overlapping_units = nx.Graph()
            for unit_index_i in np.arange(len(unit_ids)):
                for unit_index_j in np.arange(unit_index_i + 1, len(unit_ids)):
                    if _are_unit_indices_overlapping(unit_inds_to_channel_indices,
                                                     unit_index_i, unit_index_j):
                        overlapping_units.add_node(unit_index_i)
                        overlapping_units.add_node(unit_index_j)
                        overlapping_units.add_edge(unit_index_i, unit_index_j)
            cliques = [c for c in nx.find_cliques(overlapping_units)]

            all_collision_indices = np.array([], dtype=int)
            # we do this by segment to avoid border effects
            amp_scalings_collisions = np.array([])
            for segment_slice in segment_slices:
                overlapping_spikes = []
                spikes_in_segment = self.spikes[segment_slice]
                # for each unique clique, find groups of 2+ overlapping spikes
                # all_collision_indices contains all indices of spikes handled as collisions
                collision_indices_segment = np.array([], dtype=int)
                for clique in cliques:
                    indices_cliques = np.in1d(spikes_in_segment["unit_index"], clique)
                    spikes_in_clique = spikes_in_segment[indices_cliques]
                    collisions_indices = np.where(np.diff(spikes_in_clique["sample_index"]) <= delta_samples)[0]
                    if len(collisions_indices) > 0:
                        # using unique to prevent counting consecutive spikes (maybe on different channels) twice
                        collision_indices_i = np.unique(np.sort(np.concatenate((collisions_indices,
                                                                                collisions_indices + 1))))
                        collisions = spikes_in_clique[collision_indices_i]
                        collision_indices_segment = np.concatenate((collision_indices_segment,
                                                                    np.nonzero(indices_cliques)[0][collision_indices_i]))
                        overlapping_spikes.append(collisions)

                # here we split all overlapping spikes in actual "collisions" (groups of spikes that will be fit together)
                collision_spikes = []
                for overlapping_spikes_by_clique in overlapping_spikes:
                    ovs = np.split(overlapping_spikes_by_clique,
                                   np.where(np.diff(overlapping_spikes_by_clique["sample_index"]) > delta_samples)[0] + 1)
                    collision_spikes += ovs
                # now we feat each collision separately and concatenate the amps
                for collision in collision_spikes:
                    amps_c = _fit_collision(collision, we, all_templates, unit_inds_to_channel_indices,
                                            cut_out_after, cut_out_after)
                    amp_scalings_collisions = np.concatenate((amp_scalings_collisions, amps_c))
                # re-index collision indices for multi-segment
                all_collision_indices = np.concatenate((all_collision_indices,
                                                        segment_slice.start + collision_indices_segment))

            if len(all_collision_indices) != len(np.unique(all_collision_indices)):
                # Idea: make cliques non-overlapping (or allow a maximum overlap)
                raise Exception(
                    "The same collisions belongs to more than one clique! This can be due to a small number "
                    "of channels or a too large `collision_ms`. Adjust parameters or set `handle_collisions` to False"
                )
            # at last, we remove collision spikes so they are note re-fitted
            spikes_nc = np.delete(self.spikes, all_collision_indices)
            collision_mask = np.zeros_like(self.spikes, dtype=bool)
            collision_mask[all_collision_indices] = True

            # recompute segment_slices, since indices have changed!
            segment_slices = []
            for segment_index in range(we.get_num_segments()):
                i0 = np.searchsorted(spikes_nc['segment_index'], segment_index)
                i1 = np.searchsorted(spikes_nc['segment_index'], segment_index + 1)
                segment_slices.append(slice(i0, i1))
        else:
            spikes_nc = self.spikes
            collision_mask = None

        # and run
        func = _amplitude_scalings_chunk
        init_func = _init_worker_amplitude_scalings
        n_jobs = ensure_n_jobs(recording, job_kwargs.get("n_jobs", None))
        job_kwargs["n_jobs"] = n_jobs
        init_args = (
            recording,
            spikes_nc,
            all_templates,
            segment_slices,
            unit_inds_to_channel_indices,
            nbefore,
            nafter,
            cut_out_before,
            cut_out_after,
            return_scaled,
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
        (amp_scalings_nc,) = zip(*out)
        amp_scalings_nc = np.concatenate(amp_scalings_nc)

        # take care of collisions
        if handle_collisions and len(amp_scalings_nc) < len(self.spikes):
            amp_scalings = np.zeros_like(self.spikes)
            amp_scalings[~collision_mask] = amp_scalings_nc
            amp_scalings[collision_mask] = amp_scalings_collisions
        else:
            amp_scalings = amp_scalings_nc

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
    collision_ms=0.5,
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
    sparsity: ChannelSparsity
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
        By default None
    max_dense_channels: int, optional
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
        By default 16
    ms_before : float, optional
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_before is used, by default None
    ms_after : float, optional
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_after is used, by default None
    handle_collisions : bool, optional
        If True, spike collisions ()spikes which are overlapping in sparsity and in time) are fitted with 
        a multi-dimensional linear regression, by default False
    collision_ms : float, optional
        Interval in ms to detect collisions, by default 0.5
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    outputs: str
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
        sac.set_params(sparsity=sparsity, max_dense_channels=max_dense_channels, ms_before=ms_before,
                       ms_after=ms_after, handle_collisions=handle_collisions, collision_ms=collision_ms)
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
    worker_ctx["margin"] = max(nbefore, nafter)
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["unit_inds_to_channel_indices"] = unit_inds_to_channel_indices

    return worker_ctx


def _amplitude_scalings_chunk(segment_index, start_frame, end_frame, worker_ctx):
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

        # get all waveforms
        for spike in local_spikes:
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
            scalings.append(linregress_res[0])
    scalings = np.array(scalings)

    return (scalings,)


def _are_unit_indices_overlapping(unit_inds_to_channel_indices, i, j):
    if len(np.intersect1d(unit_inds_to_channel_indices[i], unit_inds_to_channel_indices[j])) > 0:
        return True
    else:
        return False


def _fit_collision(spikes, we, all_templates, unit_inds_to_channel_indices,
                   cut_out_before, cut_out_after, return_scaled=True):
    from sklearn.linear_model import LinearRegression

    recording = we.recording
    nbefore = we.nbefore
    segment_index = spikes[0]["segment_index"]
    spike_first = spikes[0]
    spike_last = spikes[-1]
    start_frame = spike_first["sample_index"] - cut_out_before
    end_frame = spike_last["sample_index"] + cut_out_after
    sample_diff = spike_last["sample_index"] - spike_first["sample_index"]

    # deal with borders
    if start_frame < 0:
        start_frame = 0
        cut_out_before = spike_first["sample_index"]
    if end_frame >= recording.get_num_samples(segment_index=segment_index):
        end_frame = recording.get_num_samples(segment_index=segment_index)
        cut_out_after = end_frame - spike_last["sample_index"]

    # construct sparsity as union between units' sparsity
    sparse_indices = np.array([], dtype="int")
    for spike in spikes:
        sparse_indices_i = unit_inds_to_channel_indices[spike["unit_index"]]
        sparse_indices = np.union1d(sparse_indices, sparse_indices_i)


    local_waveform = recording.get_traces(start_frame=start_frame,
                                          end_frame=end_frame,
                                          segment_index=segment_index,
                                          channel_ids=recording.channel_ids[sparse_indices],
                                          return_scaled=return_scaled)
    y = local_waveform.T.flatten()
    X = np.zeros((len(y), len(spikes)))
    for i, spike in enumerate(spikes):
        sample_centered = spike["sample_index"] - start_frame
        template = all_templates[spike["unit_index"]][:, sparse_indices]
        template_cut = template[nbefore - cut_out_before : nbefore + cut_out_after]
        full_template = np.zeros((cut_out_before + cut_out_after + sample_diff, len(sparse_indices)))
        full_template[sample_centered - cut_out_before:sample_centered + cut_out_after] = template_cut
        X[:, i] = full_template.T.flatten()

    reg = LinearRegression().fit(X, y)
    amps = reg.coef_
    return amps