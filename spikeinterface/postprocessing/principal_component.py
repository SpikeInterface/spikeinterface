import shutil
import pickle
import warnings
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.exceptions import NotFittedError

from spikeinterface.core.job_tools import (ChunkRecordingExecutor, ensure_n_jobs,
                                           _shared_job_kwargs_doc, fix_job_kwargs)
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

_possible_modes = ['by_channel_local', 'by_channel_global', 'concatenated']


class WaveformPrincipalComponent(BaseWaveformExtractorExtension):
    """
    Class to extract principal components from a WaveformExtractor object.
    """

    extension_name = 'principal_components'
    handle_sparsity = True

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    @classmethod
    def create(cls, waveform_extractor):
        pc = WaveformPrincipalComponent(waveform_extractor)
        return pc

    def __repr__(self):
        we = self.waveform_extractor
        clsname = self.__class__.__name__
        nseg = we.get_num_segments()
        nchan = we.get_num_channels()
        txt = f'{clsname}: {nchan} channels - {nseg} segments'
        if len(self._params) > 0:
            mode = self._params['mode']
            n_components = self._params['n_components']
            txt = txt + f'\n  mode: {mode} n_components: {n_components}'
            if self._params['sparsity'] is not None:
                txt += ' - sparse'
        return txt

    def _set_params(self, n_components=5, mode='by_channel_local',
                    whiten=True, dtype='float32', sparsity=None):
        assert mode in _possible_modes, "Invalid mode!"
        
        if self.waveform_extractor.is_sparse():
            assert sparsity is None, "WaveformExtractor is already sparse, sparsity must be None"
        
        # the sparsity in params is ONLY the injected sparsity and not the waveform_extractor one
        params = dict(n_components=int(n_components),
                      mode=str(mode),
                      whiten=bool(whiten),
                      dtype=np.dtype(dtype).str,
                      sparsity=sparsity)
        
        return params
    
    def _select_extension_data(self, unit_ids):
        new_extension_data = dict()
        for unit_id in unit_ids:
            new_extension_data[f'pca_{unit_id}'] = self._extension_data[f'pca_{unit_id}']
        for k, v in self._extension_data.items():
            if "model" in k:
                new_extension_data[k] = v
        return new_extension_data

    def get_projections(self, unit_id):
        """
        Returns the computed projections for the sampled waveforms of a unit id.

        Parameters
        ----------
        unit_id : int or str 
            The unit id to return PCA projections for

        Returns
        -------
        proj: np.array
            The PCA projections (num_waveforms, num_components, num_channels)
        """
        return self._extension_data[f'pca_{unit_id}']

    def get_pca_model(self):
        """
        Returns the scikit-learn PCA model objects.

        Returns
        -------
        pca_models: PCA object(s)
            * if mode is "by_channel_local", "pca_model" is a list of PCA model by channel
            * if mode is "by_channel_global" or "concatenated", "pca_model" is a single PCA model
        """
        mode = self._params["mode"]
        if mode == "by_channel_local":
            pca_models = []
            for chan_id in self.waveform_extractor.channel_ids:
                pca_models.append(self._extension_data[f"pca_model_{mode}_{chan_id}"])
        else:
            pca_models = self._extension_data[f"pca_model_{mode}"]
        return pca_models

    def get_all_projections(self, channel_ids=None, unit_ids=None, outputs='id'):
        """
        Returns the computed projections for the sampled waveforms of all units.

        Parameters
        ----------
        channel_ids : list, optional
            List of channel ids on which projections are computed
        unit_ids : list, optional
            List of unit ids to return projections for
        outputs: str
            * 'id': 'all_labels' contain unit ids
            * 'index': 'all_labels' contain unit indices

        Returns
        -------
        all_labels: np.array
            Array with labels (ids or indices based on 'outputs') of returned PCA projections
        all_projections: np.array
            The PCA projections (num_all_waveforms, num_components, num_channels)
        """
        if unit_ids is None:
            unit_ids = self.waveform_extractor.sorting.unit_ids

        all_labels = []  #  can be unit_id or unit_index
        all_projections = []
        for unit_index, unit_id in enumerate(unit_ids):
            proj = self.get_projections(unit_id)
            if channel_ids is not None:
                chan_inds = self.waveform_extractor.channel_ids_to_indices(channel_ids)
                proj = proj[:, :, chan_inds]
            n = proj.shape[0]
            if outputs == 'id':
                labels = np.array([unit_id] * n)
            elif outputs == 'index':
                labels = np.ones(n, dtype='int64')
                labels[:] = unit_index
            all_labels.append(labels)
            all_projections.append(proj)
        all_labels = np.concatenate(all_labels, axis=0)
        all_projections = np.concatenate(all_projections, axis=0)

        return all_labels, all_projections

    def project_new(self, new_waveforms, unit_id=None):
        """
        Projects new waveforms or traces snippets on the PC components.

        Parameters
        ----------
        new_waveforms: np.array
            Array with new waveforms to project with shape (num_waveforms, num_samples, num_channels)
        unit_id: int or str
            In case PCA is sparse and mode is by_channel_local, the unit_id of 'new_waveforms'

        Returns
        -------
        projections: np.array
            Projections of new waveforms on PCA compoents

        """
        p = self._params
        mode = p["mode"]
        sparsity = p["sparsity"]

        wfs0 = self.waveform_extractor.get_waveforms(unit_id=self.waveform_extractor.sorting.unit_ids[0])
        assert wfs0.shape[1] == new_waveforms.shape[1], \
            ("Mismatch in number of samples between waveforms used to fit the pca model and 'new_waveforms")
        num_channels = len(self.waveform_extractor.channel_ids)

        # check waveform shapes
        if sparsity is not None:
            assert unit_id is not None, \
                    "The unit_id of the new_waveforms is needed to apply the waveforms transformation"
            channel_inds = sparsity.unit_id_to_channel_indices[unit_id]
            if new_waveforms.shape[2] != len(channel_inds):
                new_waveforms = new_waveforms.copy()[:, :, channel_inds]
        else:            
            assert wfs0.shape[2] == new_waveforms.shape[2], \
                ("Mismatch in number of channels between waveforms used to fit the pca model and 'new_waveforms")
            channel_inds = np.arange(num_channels, dtype=int)

        # get channel ids and pca models
        pca_model = self.get_pca_model()
        projections = None

        if mode == "by_channel_local":
            shape = (new_waveforms.shape[0], p['n_components'], num_channels)
            projections = np.zeros(shape)
            for wf_ind, chan_ind in enumerate(channel_inds):
                pca = pca_model[chan_ind]
                projections[:, :, chan_ind] = pca.transform(new_waveforms[:, :, wf_ind])
        elif mode == "by_channel_global":
            shape = (new_waveforms.shape[0], p['n_components'], num_channels)
            projections = np.zeros(shape)
            for wf_ind, chan_ind in enumerate(channel_inds):
                projections[:, :, chan_ind] = pca_model.transform(new_waveforms[:, :, wf_ind])
        elif mode == "concatenated":
            wfs_flat = new_waveforms.reshape(new_waveforms.shape[0], -1)
            projections = pca_model.transform(wfs_flat)

        return projections

    def get_sparsity(self):
        if self.waveform_extractor.is_sparse():
            return self.waveform_extractor.sparsity
        return self._params["sparsity"]

    def _run(self, **job_kwargs):
        """
        Compute the PCs on waveforms extacted within the WaveformExtarctor.
        Projections are computed only on the waveforms sampled by the WaveformExtractor.
        
        The index of spikes come from the WaveformExtarctor.
        This will be cached in the same folder than WaveformExtarctor
        in extension subfolder.
        """
        p = self._params
        we = self.waveform_extractor
        num_chans = we.get_num_channels()

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        n_jobs = job_kwargs['n_jobs']
        progress_bar = job_kwargs['progress_bar']

        # prepare memmap files with npy
        projection_objects = {}
        unit_ids = we.unit_ids

        for unit_id in unit_ids:
            n_spike = we.get_waveforms(unit_id).shape[0]
            if p['mode'] in ('by_channel_local', 'by_channel_global'):
                shape = (n_spike, p['n_components'], num_chans)
            elif p['mode'] == 'concatenated':
                shape = (n_spike, p['n_components'])
            proj = np.zeros(shape, dtype=p['dtype'])          
            projection_objects[unit_id] = proj

        # run ...
        if p['mode'] == 'by_channel_local':
            self._run_by_channel_local(projection_objects, n_jobs, progress_bar)
        elif p['mode'] == 'by_channel_global':
            self._run_by_channel_global(projection_objects, n_jobs, progress_bar)
        elif p['mode'] == 'concatenated':
            self._run_concatenated(projection_objects, n_jobs, progress_bar)

        # add projections to extension data
        for unit_id in unit_ids:
            self._extension_data[f'pca_{unit_id}'] = projection_objects[unit_id]
            
    def get_data(self):
        """
        Get computed PCA projections.

        Returns
        -------
        all_labels : 1d np.array
            Array with all spike labels
        all_projections : 3d array
            Array with PCA projections (num_spikes, num_components, num_channels)
        """
        return self.get_all_projections()

    @staticmethod
    def get_extension_function():
        return compute_principal_components

    def run_for_all_spikes(self, file_path=None, **job_kwargs):
        """
        Project all spikes from the sorting on the PCA model.
        This is a long computation because waveform need to be extracted from each spikes.
        
        Used mainly for `export_to_phy()`
        
        PCs are exported to a .npy single file.

        Parameters
        ----------
        file_path : str or Path or None
            Path to npy file that will store the PCA projections.
            If None, output is saved in principal_components/all_pcs.npy
        {}
        """
        job_kwargs = fix_job_kwargs(job_kwargs)
        p = self._params
        we = self.waveform_extractor
        sorting = we.sorting
        assert we.has_recording(), (
            "To compute PCA projections for all spikes, the waveform extractor needs the recording"
        )
        recording = we.recording

        assert sorting.get_num_segments() == 1
        assert p['mode'] in ('by_channel_local', 'by_channel_global')

        if file_path is None:
            file_path = self.extension_folder / "all_pcs.npy"
        file_path = Path(file_path)

        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        spike_times, spike_labels = all_spikes[0]

        sparsity = self.get_sparsity()
        if sparsity is None:
            sparse_channels_indices = {unit_id: np.arange(we.get_num_channels()) for unit_id in we.unit_ids}
            max_channels_per_template = we.get_num_channels()
        else:
            sparse_channels_indices = sparsity.unit_id_to_channel_indices
            max_channels_per_template = max([chan_inds.size for chan_inds in sparse_channels_indices.values()])

        unit_channels = [sparse_channels_indices[unit_id] for unit_id in sorting.unit_ids]

        pca_model = self.get_pca_model()
        if p['mode'] in ['by_channel_global', 'concatenated']:
            pca_model = [pca_model] * recording.get_num_channels()

        # nSpikes, nFeaturesPerChannel, nPCFeatures
        # this comes from  phy template-gui
        # https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md#datasets
        shape = (spike_times.size, p['n_components'], max_channels_per_template)
        all_pcs = np.lib.format.open_memmap(filename=file_path, mode='w+', dtype='float32', shape=shape)
        all_pcs_args = dict(filename=file_path, mode='r+', dtype='float32', shape=shape)

        # and run
        func = _all_pc_extractor_chunk
        init_func = _init_work_all_pc_extractor
        n_jobs = ensure_n_jobs(recording, job_kwargs.get('n_jobs', None))
        if n_jobs == 1:
            init_args = (recording,)
        else:
            init_args = (recording.to_dict(),)
        init_args = init_args + (all_pcs_args, spike_times, spike_labels, we.nbefore, we.nafter, 
                                 unit_channels, pca_model)
        processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name='extract PCs', **job_kwargs)
        processor.run()

    def _fit_by_channel_local(self, n_jobs, progress_bar):
        from joblib import delayed, Parallel
        
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.unit_ids
        channel_ids = we.channel_ids

        # there is one PCA per channel for independent fit per channel
        pca_models = [IncrementalPCA(n_components=p['n_components'], whiten=p['whiten']) for _ in channel_ids]
        
        mode = p["mode"]
        pca_model_files = []
        tmp_folder = Path("tmp")
        for chan_ind, chan_id in enumerate(channel_ids):
            pca_model = pca_models[chan_ind]
            if n_jobs > 1:
                tmp_folder.mkdir(exist_ok=True)
                pca_model_file = tmp_folder / f"tmp_pca_model_{mode}_{chan_id}.pkl"
                with pca_model_file.open("wb") as f:
                    pickle.dump(pca_model, f)
                pca_model_files.append(pca_model_file)
        
        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs, channel_inds = self._get_sparse_waveforms(unit_id)
            if len(wfs) < p['n_components']:
                continue
            if n_jobs in (0, 1):
                for wf_ind, chan_ind in enumerate(channel_inds):
                    pca = pca_models[chan_ind]
                    pca.partial_fit(wfs[:, :, wf_ind])
            else:
                Parallel(n_jobs=n_jobs)(delayed(partial_fit_one_channel)(pca_model_files[chan_ind], wfs[:, :, wf_ind])
                                                        for wf_ind, chan_ind in enumerate(channel_inds))

        # reload the models (if n_jobs > 1)
        if n_jobs not in (0, 1):
            pca_models = []
            for chan_ind, chan_id in enumerate(channel_ids):
                pca_model_file = pca_model_files[chan_ind]
                with open(pca_model_file, 'rb') as fid:
                    pca_models.append(pickle.load(fid))
                pca_model_file.unlink()
            shutil.rmtree(tmp_folder)

        # add models to extension data
        for chan_ind, chan_id in enumerate(channel_ids):
            pca_model = pca_models[chan_ind]
            self._extension_data[f"pca_model_{mode}_{chan_id}"] = pca_model

        return pca_models

    def _run_by_channel_local(self, projection_memmap, n_jobs, progress_bar):
        """
        In this mode each PCA is "fit" and "transform" by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        unit_ids = we.unit_ids

        pca_model = self._fit_by_channel_local(n_jobs, progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        project_on_non_fitted = False
        for unit_ind, unit_id in units_loop:
            wfs, channel_inds = self._get_sparse_waveforms(unit_id)
            if wfs.size == 0:
                continue
            for wf_ind, chan_ind in enumerate(channel_inds):
                pca = pca_model[chan_ind]
                try:
                    proj = pca.transform(wfs[:, :, wf_ind])
                    projection_memmap[unit_id][:, :, chan_ind] = proj
                except NotFittedError as e:
                    # this could happen if len(wfs) is less then n_comp for a channel
                    project_on_non_fitted = True
        if project_on_non_fitted:
            warnings.warn("Projection attempted on unfitted PCA models. This could be due to a small "
                          "number of waveforms for a particular unit.")

    def _fit_by_channel_global(self, progress_bar):
        we = self.waveform_extractor
        p = self._params
        unit_ids = we.unit_ids

        # there is one unique PCA accross channels
        pca_model = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        # with 'by_channel_global' we can't parallelize over channels
        for unit_ind, unit_id in units_loop:
            wfs, _ = self._get_sparse_waveforms(unit_id)
            shape = wfs.shape
            if shape[0] * shape[2] < p['n_components']:
                continue
            # avoid loop with reshape
            wfs_concat = wfs.transpose(0, 2, 1).reshape(shape[0] * shape[2], shape[1])
            pca_model.partial_fit(wfs_concat)

        # save
        mode = p["mode"]
        self._extension_data[f"pca_model_{mode}"] = pca_model

        return pca_model

    def _run_by_channel_global(self, projection_objects, n_jobs, progress_bar):
        """
        In this mode there is one "fit" for all channels.
        The transform is applied by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        unit_ids = we.unit_ids

        pca_model = self._fit_by_channel_global(progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        # with 'by_channel_global' we can't parallelize over channels
        for unit_ind, unit_id in units_loop:
            wfs, channel_inds = self._get_sparse_waveforms(unit_id)
            if wfs.size == 0:
                continue
            for wf_ind, chan_ind in enumerate(channel_inds):
                proj = pca_model.transform(wfs[:, :, wf_ind])
                projection_objects[unit_id][:, :, chan_ind] = proj
                
    def _fit_concatenated(self, progress_bar):
        we = self.waveform_extractor
        p = self._params
        unit_ids = we.unit_ids
        
        sparsity = self.get_sparsity()
        if sparsity is not None:
            sparsity0 = sparsity.unit_id_to_channel_indices[unit_ids[0]]
            assert all(len(chans) == len(sparsity0) for u, chans in sparsity.unit_id_to_channel_indices.items()), \
                "When using sparsity in concatenated mode, make sure each unit has the same number of sparse channels"
        
        # there is one unique PCA accross channels
        pca_model = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs, _ = self._get_sparse_waveforms(unit_id)
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            if len(wfs_flat) < p['n_components']:
                continue
            pca_model.partial_fit(wfs_flat)

        # save
        mode = p["mode"]
        self._extension_data[f"pca_model_{mode}"] = pca_model

        return pca_model

    def _run_concatenated(self, projection_objects, n_jobs, progress_bar):
        """
        In this mode the waveforms are concatenated and there is
        a global fit_transform at once.
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.unit_ids

        # there is one unique PCA accross channels
        pca_model = self._fit_concatenated(progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs, _ = self._get_sparse_waveforms(unit_id)
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            proj = pca_model.transform(wfs_flat)
            projection_objects[unit_id][:, :] = proj

    def _get_sparse_waveforms(self, unit_id):
        # get waveforms : dense or sparse
        we = self.waveform_extractor
        sparsity = self._params['sparsity']
        if we.is_sparse():
            # natural sparsity
            wfs = we.get_waveforms(unit_id, lazy=False)
            channel_inds = we.sparsity.unit_id_to_channel_indices[unit_id]
        elif sparsity is not None:
            # injected sparsity
            wfs = self.waveform_extractor.get_waveforms(unit_id, sparsity=sparsity, lazy=False)
            channel_inds = sparsity.unit_id_to_channel_indices[unit_id]
        else:
            # dense
            wfs = self.waveform_extractor.get_waveforms(unit_id, sparsity=None, lazy=False)
            channel_inds = np.arange(we.channel_ids.size, dtype=int)
        return wfs, channel_inds

def _all_pc_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx['recording']
    all_pcs = worker_ctx['all_pcs']
    spike_times = worker_ctx['spike_times']
    spike_labels = worker_ctx['spike_labels']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    unit_channels = worker_ctx['unit_channels']
    pca_model = worker_ctx['pca_model']

    seg_size = recording.get_num_samples(segment_index=segment_index)

    i0 = np.searchsorted(spike_times, start_frame)
    i1 = np.searchsorted(spike_times, end_frame)

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

        wf = traces[st - start - nbefore:st - start + nafter, :]

        unit_index = spike_labels[i]
        chan_inds = unit_channels[unit_index]

        for c, chan_ind in enumerate(chan_inds):
            w = wf[:, chan_ind]
            if w.size > 0:
                w = w[None, :]
                all_pcs[i, :, c] = pca_model[chan_ind].transform(w)


def _init_work_all_pc_extractor(recording, all_pcs_args, spike_times, spike_labels, nbefore, nafter, unit_channels,
                                pca_model):
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    worker_ctx['all_pcs'] = np.lib.format.open_memmap(**all_pcs_args)
    worker_ctx['spike_times'] = spike_times
    worker_ctx['spike_labels'] = spike_labels
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['unit_channels'] = unit_channels
    worker_ctx['pca_model'] = pca_model

    return worker_ctx


WaveformPrincipalComponent.run_for_all_spikes.__doc__ = WaveformPrincipalComponent.run_for_all_spikes.__doc__.format(
    _shared_job_kwargs_doc)

WaveformExtractor.register_extension(WaveformPrincipalComponent)


def compute_principal_components(waveform_extractor, load_if_exists=False,
                                 n_components=5, mode='by_channel_local', sparsity=None,
                                 whiten=True, dtype='float32', **job_kwargs):
    """
    Compute PC scores from waveform extractor. The PCA projections are pre-computed only
    on the sampled waveforms available from the WaveformExtractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    load_if_exists: bool
        If True and pc scores are already in the waveform extractor folders, pc scores are loaded and not recomputed.
    n_components: int
        Number of components fo PCA - default 5
    mode: str
        - 'by_channel_local': a local PCA is fitted for each channel (projection by channel)
        - 'by_channel_global': a global PCA is fitted for all channels (projection by channel)
        - 'concatenated': channels are concatenated and a global PCA is fitted
    sparsity: ChannelSparsity or None
        The sparsity to apply to waveforms.
        If waveform_extractor is already sparse, the default sparsity will be used.
    whiten: bool
        If True, waveforms are pre-whitened
    dtype: dtype
        Dtype of the pc scores (default float32)
    n_jobs: int
        Number of jobs used to fit the PCA model (if mode is 'by_channel_local') - default 1
    progress_bar: bool
        If True, a progress bar is shown - default False

    Returns
    -------
    pc: WaveformPrincipalComponent
        The waveform principal component object

    Examples
    --------
    >>> we = si.extract_waveforms(recording, sorting, folder='waveforms')
    >>> pc = st.compute_principal_components(we, n_components=3, mode='by_channel_local')
    >>> # get pre-computed projections for unit_id=1
    >>> projections = pc.get_projections(unit_id=1)
    >>> # get all pre-computed projections and labels
    >>> all_projections, all_labels = pc.get_all_projections()
    >>> # retrieve fitted pca model(s)
    >>> pca_model = pc.get_pca_model()
    >>> # compute projections on new waveforms
    >>> proj_new = pc.project_new(new_waveforms)
    >>> # run for all spikes in the SortingExtractor
    >>> pc.run_for_all_spikes(file_path="all_pca_projections.npy")
    """
    if load_if_exists and waveform_extractor.is_extension(WaveformPrincipalComponent.extension_name):
        pc = waveform_extractor.load_extension(WaveformPrincipalComponent.extension_name)
    else:
        pc = WaveformPrincipalComponent.create(waveform_extractor)
        pc.set_params(n_components=n_components, mode=mode, whiten=whiten, dtype=dtype,
                      sparsity=sparsity)
        pc.run(**job_kwargs)

    return pc


def partial_fit_one_channel(pca_file, wf_chan):
    with open(pca_file, 'rb') as fid:
        pca_model = pickle.load(fid)
    pca_model.partial_fit(wf_chan)
    with pca_file.open("wb") as f:
        pickle.dump(pca_model, f)
