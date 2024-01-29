import shutil
import pickle
import warnings
import tempfile
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np

from spikeinterface.core.sortingresult import register_result_extension, ResultExtension

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, fix_job_kwargs
# from spikeinterface.core.globals import get_global_tmp_folder

_possible_modes = ["by_channel_local", "by_channel_global", "concatenated"]


# TODO handle extra sparsity

class ComputePrincipalComponents(ResultExtension):
    """
    Compute PC scores from waveform extractor. The PCA projections are pre-computed only
    on the sampled waveforms available from the WaveformExtractor.

    Parameters
    ----------
    sorting_result: SortingResult
        A SortingResult object
    n_components: int, default: 5
        Number of components fo PCA
    mode: "by_channel_local" | "by_channel_global" | "concatenated", default: "by_channel_local"
        The PCA mode:
            - "by_channel_local": a local PCA is fitted for each channel (projection by channel)
            - "by_channel_global": a global PCA is fitted for all channels (projection by channel)
            - "concatenated": channels are concatenated and a global PCA is fitted
    sparsity: ChannelSparsity or None, default: None
        The sparsity to apply to waveforms.
        If sorting_result is already sparse, the default sparsity will be used
    whiten: bool, default: True
        If True, waveforms are pre-whitened
    dtype: dtype, default: "float32"
        Dtype of the pc scores

    Examples
    --------
    >>> we = si.extract_waveforms(recording, sorting, folder='waveforms')
    >>> pc = st.compute_principal_components(we, n_components=3, mode='by_channel_local')
    >>> # get pre-computed projections for unit_id=1
    >>> projections = pc.get_projections(unit_id=1)
    >>> # retrieve fitted pca model(s)
    >>> pca_model = pc.get_pca_model()
    >>> # compute projections on new waveforms
    >>> proj_new = pc.project_new(new_waveforms)
    >>> # run for all spikes in the SortingExtractor
    >>> pc.run_for_all_spikes(file_path="all_pca_projections.npy")
    """

    extension_name = "principal_components"
    depend_on = ["waveforms", ]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True

    def __init__(self, sorting_result):
        ResultExtension.__init__(self, sorting_result)

    def _set_params(
        self, n_components=5, mode="by_channel_local", whiten=True, dtype="float32", sparsity=None, 
    ):
        assert mode in _possible_modes, "Invalid mode!"

        if sparsity is not None:
            # TODO alessio:  implement local sparsity or not ??
            raise NotImplementedError

        # the sparsity in params is ONLY the injected sparsity and not the sorting_result one
        params = dict(
            n_components=n_components,
            mode=mode,
            whiten=whiten,
            dtype=np.dtype(dtype),
            # sparsity=sparsity,
            # tmp_folder=tmp_folder,
        )
        return params

    def _select_extension_data(self, unit_ids):

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))
        spikes = self.sorting_result.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        keep_spike_mask = np.isin(some_spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["pca_projection"] = self.data["pca_projection"][keep_spike_mask, :, :]
        # one or several model
        for k, v in self.data.items():
            if "model" in k:
                new_data[k] = v
        return new_data


    def get_projections(self, unit_id, sparse=False):
        """
        Returns the computed projections for the sampled waveforms of a unit id.

        Parameters
        ----------
        unit_id : int or str
            The unit id to return PCA projections for
        sparse: bool, default: False
            If True, and sparsity is not None, only projections on sparse channels are returned.

        Returns
        -------
        projections: np.array
            The PCA projections (num_waveforms, num_components, num_channels).
            In case sparsity is used, only the projections on sparse channels are returned.
        """
        projections = self.data[f"pca_{unit_id}"]
        mode = self.params["mode"]
        if mode in ("by_channel_local", "by_channel_global") and sparse:
            sparsity = self.get_sparsity()
            if sparsity is not None:
                projections = projections[:, :, sparsity.unit_id_to_channel_indices[unit_id]]
        return projections

    def get_pca_model(self):
        """
        Returns the scikit-learn PCA model objects.

        Returns
        -------
        pca_models: PCA object(s)
            * if mode is "by_channel_local", "pca_model" is a list of PCA model by channel
            * if mode is "by_channel_global" or "concatenated", "pca_model" is a single PCA model
        """
        mode = self.params["mode"]
        if mode == "by_channel_local":
            pca_models = []
            for chan_id in self.sorting_result.channel_ids:
                pca_models.append(self.data[f"pca_model_{mode}_{chan_id}"])
        else:
            pca_models = self.data[f"pca_model_{mode}"]
        return pca_models

    # def get_all_projections(self, channel_ids=None, unit_ids=None, outputs="id"):
    #     """
    #     Returns the computed projections for the sampled waveforms of all units.

    #     Parameters
    #     ----------
    #     channel_ids : list, default: None
    #         List of channel ids on which projections are computed
    #     unit_ids : list, default: None
    #         List of unit ids to return projections for
    #     outputs: str
    #         * "id": "all_labels" contain unit ids
    #         * "index": "all_labels" contain unit indices

    #     Returns
    #     -------
    #     all_labels: np.array
    #         Array with labels (ids or indices based on "outputs") of returned PCA projections
    #     all_projections: np.array
    #         The PCA projections (num_all_waveforms, num_components, num_channels)
    #     """
    #     if unit_ids is None:
    #         unit_ids = self.sorting_result.sorting.unit_ids

    #     all_labels = []  #  can be unit_id or unit_index
    #     all_projections = []
    #     for unit_index, unit_id in enumerate(unit_ids):
    #         proj = self.get_projections(unit_id, sparse=False)
    #         if channel_ids is not None:
    #             chan_inds = self.sorting_result.chanpca_projectionnel_ids_to_indices(channel_ids)
    #             proj = proj[:, :, chan_inds]
    #         n = proj.shape[0]
    #         if outputs == "id":
    #             labels = np.array([unit_id] * n)
    #         elif outputs == "index":
    #             labels = np.ones(n, dtype="int64")
    #             labels[:] = unit_index
    #         all_labels.append(labels)
    #         all_projections.append(proj)
    #     all_labels = np.concatenate(all_labels, axis=0)
    #     all_projections = np.concatenate(all_projections, axis=0)

    #     return all_labels, all_projections

    def project_new(self, new_spikes, new_waveforms, progress_bar=True):
        """
        Projects new waveforms or traces snippets on the PC components.

        Parameters
        ----------
        new_spikes: np.array
            The spikes vector associated to the waveforms buffer. This is need need to get the sparsity spike per spike.
        new_waveforms: np.array
            Array with new waveforms to project with shape (num_waveforms, num_samples, num_channels)

        Returns
        -------
        new_projections: np.array
            Projections of new waveforms on PCA compoents

        """
        pca_model = self.get_pca_model()
        new_projections = self._transform_waveforms( new_spikes, new_waveforms, pca_model, progress_bar=progress_bar)
        return new_projections


    def get_sparsity(self):
        if self.sorting_result.is_sparse():
            return self.sorting_result.sparsity
        return self.params["sparsity"]

    def _run(self, **job_kwargs):
        """
        Compute the PCs on waveforms extacted within the by ComputeWaveforms.
        Projections are computed only on the waveforms sampled by the SortingResult.
        """
        p = self.params
        mode = p["mode"]

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        n_jobs = job_kwargs["n_jobs"]
        progress_bar = job_kwargs["progress_bar"]

        # fit model/models
        # TODO : make parralel  for by_channel_global and concatenated
        if mode == "by_channel_local":
            pca_models = self._fit_by_channel_local(n_jobs, progress_bar)
            for chan_ind, chan_id in enumerate(self.sorting_result.channel_ids):
                self.data[f"pca_model_{mode}_{chan_id}"] = pca_models[chan_ind]
            pca_model = pca_models
        elif mode == "by_channel_global":
            pca_model = self._fit_by_channel_global(progress_bar)
            self.data[f"pca_model_{mode}"] = pca_model
        elif mode == "concatenated":
            pca_model = self._fit_concatenated(progress_bar)
            self.data[f"pca_model_{mode}"] = pca_model


        # transform
        waveforms_ext = self.sorting_result.get_extension("waveforms")
        some_waveforms = waveforms_ext.data["waveforms"]
        spikes = self.sorting_result.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        
        pca_projection = self._transform_waveforms(some_spikes, some_waveforms, pca_model, progress_bar)

        self.data["pca_projection"] = pca_projection


    def _get_data(self):
        return self.data["pca_projection"]

    # @staticmethod
    # def get_extension_function():
    #     return compute_principal_components

    # def run_for_all_spikes(self, file_path=None, **job_kwargs):
    #     """
    #     Project all spikes from the sorting on the PCA model.
    #     This is a long computation because waveform need to be extracted from each spikes.

    #     Used mainly for `export_to_phy()`

    #     PCs are exported to a .npy single file.

    #     Parameters
    #     ----------
    #     file_path : str or Path or None
    #         Path to npy file that will store the PCA projections.
    #         If None, output is saved in principal_components/all_pcs.npy
    #     {}
    #     """

    #     job_kwargs = fix_job_kwargs(job_kwargs)
    #     p = self.params
    #     we = self.sorting_result
    #     sorting = we.sorting
    #     assert (
    #         we.has_recording()
    #     ), "To compute PCA projections for all spikes, the waveform extractor needs the recording"
    #     recording = we.recording

    #     assert sorting.get_num_segments() == 1
    #     assert p["mode"] in ("by_channel_local", "by_channel_global")

    #     if file_path is None:
    #         file_path = self.extension_folder / "all_pcs.npy"
    #     file_path = Path(file_path)


    #     sparsity = self.get_sparsity()
    #     if sparsity is None:
    #         sparse_channels_indices = {unit_id: np.arange(we.get_num_channels()) for unit_id in we.unit_ids}
    #         max_channels_per_template = we.get_num_channels()
    #     else:
    #         sparse_channels_indices = sparsity.unit_id_to_channel_indices
    #         max_channels_per_template = max([chan_inds.size for chan_inds in sparse_channels_indices.values()])

    #     unit_channels = [sparse_channels_indices[unit_id] for unit_id in sorting.unit_ids]

    #     pca_model = self.get_pca_model()
    #     if p["mode"] in ["by_channel_global", "concatenated"]:
    #         pca_model = [pca_model] * recording.get_num_channels()

    #     num_spikes = sorting.to_spike_vector().size
    #     shape = (num_spikes, p["n_components"], max_channels_per_template)
    #     all_pcs = np.lib.format.open_memmap(filename=file_path, mode="w+", dtype="float32", shape=shape)
    #     all_pcs_args = dict(filename=file_path, mode="r+", dtype="float32", shape=shape)

    #     # and run
    #     func = _all_pc_extractor_chunk
    #     init_func = _init_work_all_pc_extractor
    #     init_args = (
    #         recording,
    #         sorting.to_multiprocessing(job_kwargs["n_jobs"]),
    #         all_pcs_args,
    #         we.nbefore,
    #         we.nafter,
    #         unit_channels,
    #         pca_model,
    #     )
    #     processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name="extract PCs", **job_kwargs)
    #     processor.run()

    def _fit_by_channel_local(self, n_jobs, progress_bar):
        from sklearn.decomposition import IncrementalPCA
        from concurrent.futures import ProcessPoolExecutor

        p = self.params

        unit_ids = self.sorting_result.unit_ids
        channel_ids = self.sorting_result.channel_ids
        # there is one PCA per channel for independent fit per channel
        pca_models = [IncrementalPCA(n_components=p["n_components"], whiten=p["whiten"]) for _ in channel_ids]

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs, channel_inds, _ = self._get_sparse_waveforms(unit_id)
            if len(wfs) < p["n_components"]:
                continue
            if n_jobs in (0, 1):
                for wf_ind, chan_ind in enumerate(channel_inds):
                    pca = pca_models[chan_ind]
                    pca.partial_fit(wfs[:, :, wf_ind])
            else:
                # parallel
                items = [(chan_ind, pca_models[chan_ind], wfs[:, :, wf_ind]) for wf_ind, chan_ind in enumerate(channel_inds)]
                n_jobs = min(n_jobs, len(items))

                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    results = executor.map(partial_fit_one_channel, items)
                    for chan_ind, pca_model_updated in results:
                        pca_models[chan_ind] = pca_model_updated

        return pca_models

    def _fit_by_channel_global(self, progress_bar):
        # we = self.sorting_result
        p = self.params
        # unit_ids = we.unit_ids
        unit_ids = self.sorting_result.unit_ids

        # there is one unique PCA accross channels
        from sklearn.decomposition import IncrementalPCA

        pca_model = IncrementalPCA(n_components=p["n_components"], whiten=p["whiten"])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        # with 'by_channel_global' we can't parallelize over channels
        for unit_ind, unit_id in units_loop:
            wfs, _, _ = self._get_sparse_waveforms(unit_id)
            shape = wfs.shape
            if shape[0] * shape[2] < p["n_components"]:
                continue
            # avoid loop with reshape
            wfs_concat = wfs.transpose(0, 2, 1).reshape(shape[0] * shape[2], shape[1])
            pca_model.partial_fit(wfs_concat)


        return pca_model
        
    def _fit_concatenated(self, progress_bar):

        p = self.params
        unit_ids = self.sorting_result.unit_ids

        assert self.sorting_result.sparsity is None, "For mode 'concatenated' waveforms need to be dense"

        # there is one unique PCA accross channels
        from sklearn.decomposition import IncrementalPCA

        pca_model = IncrementalPCA(n_components=p["n_components"], whiten=p["whiten"])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs, _, _ = self._get_sparse_waveforms(unit_id)
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            if len(wfs_flat) < p["n_components"]:
                continue
            pca_model.partial_fit(wfs_flat)

        return pca_model
    

    def _transform_waveforms(self, spikes, waveforms, pca_model, progress_bar):
        # transform a waveforms buffer
        # used by _run() and project_new()

        from sklearn.exceptions import NotFittedError

        mode = self.params["mode"]

        # prepare buffer
        n_components = self.params["n_components"]
        if mode in ("by_channel_local", "by_channel_global"):
            shape = (waveforms.shape[0], n_components, waveforms.shape[2])
        elif mode == "concatenated":
            shape = (waveforms.shape[0], n_components)
        pca_projection = np.zeros(shape, dtype="float32")

        unit_ids = self.sorting_result.unit_ids

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        if mode == "by_channel_local":
            # in this case the model is a list of model
            pca_models = pca_model

            project_on_non_fitted = False
            for unit_ind, unit_id in units_loop:
                wfs, channel_inds, spike_mask = self._get_slice_waveforms(unit_id, spikes, waveforms)
                if wfs.size == 0:
                    continue
                for wf_ind, chan_ind in enumerate(channel_inds):
                    pca_model = pca_models[chan_ind]
                    try:
                        proj = pca_model.transform(wfs[:, :, wf_ind])
                        pca_projection[:, :, wf_ind][spike_mask, : ] = proj
                    except NotFittedError as e:
                        # this could happen if len(wfs) is less then n_comp for a channel
                        project_on_non_fitted = True
            if project_on_non_fitted:
                warnings.warn(
                    "Projection attempted on unfitted PCA models. This could be due to a small "
                    "number of waveforms for a particular unit."
                )
        elif mode == "by_channel_global":
            # with 'by_channel_global' we can't parallelize over channels
            for unit_ind, unit_id in units_loop:
                wfs, channel_inds, spike_mask = self._get_slice_waveforms(unit_id, spikes, waveforms)
                if wfs.size == 0:
                    continue
                for wf_ind, chan_ind in enumerate(channel_inds):
                    proj = pca_model.transform(wfs[:, :, wf_ind])
                    pca_projection[:, :, wf_ind][spike_mask, : ] = proj
        elif mode == "concatenated":
            for unit_ind, unit_id in units_loop:
                wfs, channel_inds, spike_mask = self._get_slice_waveforms(unit_id, spikes, waveforms)
                wfs_flat = wfs.reshape(wfs.shape[0], -1)
                proj = pca_model.transform(wfs_flat)
                pca_projection[spike_mask, :] = proj
        
        return pca_projection

    def _get_slice_waveforms(self, unit_id, spikes, waveforms):
        # slice by mask waveforms from one unit 

        unit_index = self.sorting_result.sorting.id_to_index(unit_id)
        spike_mask = spikes["unit_index"] == unit_index
        wfs = waveforms[spike_mask, :, :]
        
        sparsity = self.sorting_result.sparsity
        if sparsity is not None:
            channel_inds = sparsity.unit_id_to_channel_indices[unit_id]
            wfs = wfs[:, :, :channel_inds.size]
        else:
            channel_inds = np.arange(self.sorting_result.channel_ids.size, dtype=int)

        return wfs, channel_inds, spike_mask

    def _get_sparse_waveforms(self, unit_id):
        # get waveforms + channel_inds: dense or sparse
        waveforms_ext = self.sorting_result.get_extension("waveforms")
        some_waveforms = waveforms_ext.data["waveforms"]

        spikes = self.sorting_result.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]

        return self._get_slice_waveforms(unit_id, some_spikes, some_waveforms)


def _all_pc_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    all_pcs = worker_ctx["all_pcs"]
    spike_times = worker_ctx["spike_times"]
    spike_labels = worker_ctx["spike_labels"]
    nbefore = worker_ctx["nbefore"]
    nafter = worker_ctx["nafter"]
    unit_channels = worker_ctx["unit_channels"]
    pca_model = worker_ctx["pca_model"]

    seg_size = recording.get_num_samples(segment_index=segment_index)

    i0, i1 = np.searchsorted(spike_times, [start_frame, end_frame])

    if i0 != i1:
        # protect from spikes on border :  spike_time<0 or spike_time>seg_size
        # usefull only when max_spikes_per_unit is not None
        # waveform will not be extracted and a zeros will be left in the memmap file
        while (spike_times[i0] - nbefore) < 0 and (i0 != i1):
            i0 = i0 + 1
        while (spike_times[i1 - 1] + nafter) > seg_size and (i0 != i1):
            i1 = i1 - 1

    if i0 == i1:
        return

    start = int(spike_times[i0] - nbefore)
    end = int(spike_times[i1 - 1] + nafter)
    traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index)

    for i in range(i0, i1):
        st = spike_times[i]
        if st - start - nbefore < 0:
            continue
        if st - start + nafter > traces.shape[0]:
            continue

        wf = traces[st - start - nbefore : st - start + nafter, :]

        unit_index = spike_labels[i]
        chan_inds = unit_channels[unit_index]

        for c, chan_ind in enumerate(chan_inds):
            w = wf[:, chan_ind]
            if w.size > 0:
                w = w[None, :]
                all_pcs[i, :, c] = pca_model[chan_ind].transform(w)


def _init_work_all_pc_extractor(recording, sorting, all_pcs_args, nbefore, nafter, unit_channels, pca_model):
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor

        recording = load_extractor(recording)
    worker_ctx["recording"] = recording
    worker_ctx["sorting"] = sorting

    spikes = sorting.to_spike_vector(concatenated=False)
    # This is the first segment only
    spikes = spikes[0]
    spike_times = spikes["sample_index"]
    spike_labels = spikes["unit_index"]

    worker_ctx["all_pcs"] = np.lib.format.open_memmap(**all_pcs_args)
    worker_ctx["spike_times"] = spike_times
    worker_ctx["spike_labels"] = spike_labels
    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["unit_channels"] = unit_channels
    worker_ctx["pca_model"] = pca_model

    return worker_ctx


# WaveformPrincipalComponent.run_for_all_spikes.__doc__ = WaveformPrincipalComponent.run_for_all_spikes.__doc__.format(
#     _shared_job_kwargs_doc
# )

register_result_extension(ComputePrincipalComponents)
compute_principal_components = ComputePrincipalComponents.function_factory()


# def partial_fit_one_channel(args):
#     pca_file, wf_chan = args
#     with open(pca_file, "rb") as fid:
#         pca_model = pickle.load(fid)
#     pca_model.partial_fit(wf_chan)
#     with pca_file.open("wb") as f:
#         pickle.dump(pca_model, f)

def partial_fit_one_channel(args):
    chan_ind, pca_model, wf_chan = args
    pca_model.partial_fit(wf_chan)
    return chan_ind, pca_model

