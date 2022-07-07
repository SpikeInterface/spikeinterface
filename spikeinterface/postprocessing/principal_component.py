import shutil
import json
import pickle
import warnings
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np

from sklearn.decomposition import IncrementalPCA

from spikeinterface.core.core_tools import check_json
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
from .template_tools import get_template_channel_sparsity

from spikeinterface.core.job_tools import _shared_job_kwargs_doc

_possible_modes = ['by_channel_local', 'by_channel_global', 'concatenated']


class WaveformPrincipalComponent(BaseWaveformExtractorExtension):
    """
    Class to extract principal components from a WaveformExtractor object.
    """

    extension_name = 'principal_components'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        self._pca_model = None

    def _specific_load_from_folder(self):
        self.get_pca_model()

    @classmethod
    def create(cls, waveform_extractor):
        pc = WaveformPrincipalComponent(waveform_extractor)
        return pc

    def __repr__(self):
        we = self.waveform_extractor
        clsname = self.__class__.__name__
        nseg = we.recording.get_num_segments()
        nchan = we.recording.get_num_channels()
        txt = f'{clsname}: {nchan} channels - {nseg} segments'
        if len(self._params) > 0:
            mode = self._params['mode']
            n_components = self._params['n_components']
            txt = txt + f'\n  mode:{mode} n_components:{n_components}'
        return txt

    def _reset(self):
        self._pca_model = None

    def _set_params(self, n_components=5, mode='by_channel_local',
                   whiten=True, dtype='float32'):

        assert mode in _possible_modes, "Invalid mode!"
        
        params = dict(n_components=int(n_components),
                      mode=str(mode),
                      whiten=bool(whiten),
                      dtype=np.dtype(dtype).str)
        
        return params
    
    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # populate folder
        pca_files = [f for f in (
            self.extension_folder).iterdir() if f.suffix == ".npy"]
        pca_model_files = [f for f in (
            self.extension_folder).iterdir() if f.suffix == ".pkl"]
        for unit in unit_ids:
            for pca_file in pca_files:
                if f"pca_{unit}.npy" in pca_file.name:
                    shutil.copyfile(pca_file, new_waveforms_folder / 
                                    self.extension_name / pca_file.name)
        for pca_model_file in pca_model_files:
            shutil.copyfile(pca_model_file, new_waveforms_folder /
                            self.extension_name / pca_model_file.name)
                

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
        proj_file = self.extension_folder / f'pca_{unit_id}.npy'
        proj = np.load(proj_file)
        return proj

    def load_pca_model(self):
        """
        Load PCA model from folder.
        """
        mode = self._params["mode"]
        if mode == "by_channel_local":
            pca_model = []
            for chan_ind, chan_id in enumerate(self.waveform_extractor.recording.channel_ids):
                pca_file = self.extension_folder / f"pca_model_{mode}_{chan_id}.pkl"
                if not pca_file.is_file() and chan_ind == 0:
                    _ = self._fit_by_channel_local()
                with open(pca_file, 'rb') as fid:
                    pca = pickle.load(fid)
                pca_model.append(pca)
        elif mode == "by_channel_global":
            pca_file = self.extension_folder / f"pca_model_{mode}.pkl"
            if not pca_file.is_file():
                _ = self._fit_by_channel_global()
            with open(pca_file, 'rb') as fid:
                pca_model = pickle.load(fid)
        elif mode == "concatenated":
            pca_file = self.extension_folder / f"pca_model_{mode}.pkl"
            if not pca_file.is_file():
                _ = self._fit_concatenated()
            with open(pca_file, 'rb') as fid:
                pca_model = pickle.load(fid)
        self._pca_model = pca_model

    def get_pca_model(self):
        """
        Returns the scikit-learn PCA model objects.

        Returns
        -------
        pca_model: PCA object(s)
            * if mode is "by_channel_local", "pca_model" is a list of PCA model by channel
            * if mode is "by_channel_global" or "concatenated", "pca_model" is a single PCA model
        """
        if self._pca_model is None:
            self.load_pca_model()
        return self._pca_model

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
        recording = self.waveform_extractor.recording

        if unit_ids is None:
            unit_ids = self.waveform_extractor.sorting.unit_ids

        all_labels = []  #  can be unit_id or unit_index
        all_projections = []
        for unit_index, unit_id in enumerate(unit_ids):
            proj = self.get_projections(unit_id)
            if channel_ids is not None:
                chan_inds = recording.ids_to_indices(channel_ids)
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

    def project_new(self, new_waveforms):
        """
        Projects new waveforms or traces snippets on the PC components.

        Parameters
        ----------
        new_waveforms: np.array
            Array with new waveforms to project with shape (num_waveforms, num_samples, num_channels)

        Returns
        -------
        projections: np.array
            Projections of new waveforms on PCA compoents

        """
        p = self._params
        mode = p["mode"]

        # check waveform shapes
        wfs0 = self.waveform_extractor.get_waveforms(unit_id=self.waveform_extractor.sorting.unit_ids[0])
        assert wfs0.shape[1] == new_waveforms.shape[1], "Mismatch in number of samples between waveforms used to fit" \
                                                        "the pca model and 'new_waveforms"
        assert wfs0.shape[2] == new_waveforms.shape[2], "Mismatch in number of channels between waveforms used to fit" \
                                                        "the pca model and 'new_waveforms"

        # get channel ids and pca models
        channel_ids = self.waveform_extractor.recording.channel_ids
        pca_model = self.get_pca_model()

        projections = None
        if mode == "by_channel_local":
            shape = (new_waveforms.shape[0], p['n_components'], len(channel_ids))
            projections = np.zeros(shape)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = pca_model[chan_ind]
                projections[:, :, chan_ind] = pca.transform(new_waveforms[:, :, chan_ind])
        elif mode == "by_channel_global":
            shape = (new_waveforms.shape[0], p['n_components'], len(channel_ids))
            projections = np.zeros(shape)
            for chan_ind, chan_id in enumerate(channel_ids):
                projections[:, :, chan_ind] = pca_model.transform(new_waveforms[:, :, chan_ind])
        elif mode == "concatenated":
            wfs_flat = new_waveforms.reshape(new_waveforms.shape[0], -1)
            projections = pca_model.transform(wfs_flat)

        return projections

    def run(self, n_jobs=1, progress_bar=False):
        """
        Compute the PCs on waveforms extacted within the WaveformExtarctor.
        Projections are computed only on the waveforms sampled by the WaveformExtractor.
        
        The index of spikes come from the WaveformExtarctor.
        This will be cached in the same folder than WaveformExtarctor
        in extension subfolder.
        """
        p = self._params
        we = self.waveform_extractor
        num_chans = we.recording.get_num_channels()

        # prepare memmap files with npy
        projection_memmap = {}
        unit_ids = we.sorting.unit_ids

        for unit_id in unit_ids:
            n_spike = we.get_waveforms(unit_id).shape[0]
            projection_file = self.extension_folder / f'pca_{unit_id}.npy'
            if p['mode'] in ('by_channel_local', 'by_channel_global'):
                shape = (n_spike, p['n_components'], num_chans)
            elif p['mode'] == 'concatenated':
                shape = (n_spike, p['n_components'])
            proj = np.zeros(shape, dtype=p['dtype'])
            np.save(projection_file, proj)
            comp = np.load(str(projection_file), mmap_mode='r+')
            projection_memmap[unit_id] = comp

        # run ...
        if p['mode'] == 'by_channel_local':
            self._run_by_channel_local(projection_memmap, n_jobs, progress_bar)
        elif p['mode'] == 'by_channel_global':
            self._run_by_channel_global(projection_memmap, n_jobs, progress_bar)
        elif p['mode'] == 'concatenated':
            self._run_concatenated(projection_memmap, n_jobs, progress_bar)
            
    def get_data(self):
        return self.get_all_projections()

    def run_for_all_spikes(self, file_path, max_channels_per_template=16, peak_sign='neg',
                           **job_kwargs):
        """
        Project all spikes from the sorting on the PCA model.
        This is a long computation because waveform need to be extracted from each spikes.
        
        Used mainly for `export_to_phy()`
        
        PCs are exported to a .npy single file.

        Parameters
        ----------
        file_path : str or Path
            Path to npy file that will store the PCA projections
        max_channels_per_template : int, optionl
            Maximum number of best channels to compute PCA projections on
        peak_sign : str, optional
            Peak sign to get best channels ('neg', 'pos', 'both'), by default 'neg'
        {}
        """
        p = self._params
        we = self.waveform_extractor
        sorting = we.sorting
        recording = we.recording

        assert sorting.get_num_segments() == 1
        assert p['mode'] in ('by_channel_local', 'by_channel_global')

        file_path = Path(file_path)

        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        spike_times, spike_labels = all_spikes[0]

        max_channels_per_template = min(max_channels_per_template, we.recording.get_num_channels())

        best_channels_index = get_template_channel_sparsity(we, outputs="index", peak_sign=peak_sign,
                                                            num_channels=max_channels_per_template)

        unit_channels = [best_channels_index[unit_id] for unit_id in sorting.unit_ids]

        pca_model = self.get_pca_model()
        if p['mode'] in ['by_channel_global', 'concatenated']:
            pca_model = [pca_model] * recording.get_num_channels()

        # nSpikes, nFeaturesPerChannel, nPCFeatures
        # this come from  phy template-gui
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

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        # there is one PCA per channel for independent fit per channel
        pca_model = [IncrementalPCA(n_components=p['n_components'], whiten=p['whiten']) for _ in channel_ids]
        
        mode = p["mode"]
        pca_model_files = []
        for chan_ind, chan_id in enumerate(channel_ids):
            pca = pca_model[chan_ind]
            pca_model_file = self.extension_folder / f"pca_model_{mode}_{chan_id}.pkl"
            with pca_model_file.open("wb") as f:
                pickle.dump(pca, f)
            pca_model_files.append(pca_model_file)
        
        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            if len(wfs) < p['n_components']:
                continue
            # try to paralleliza this
            if n_jobs in (0, 1):
                for chan_ind, chan_id in enumerate(channel_ids):
                    pca = pca_model[chan_ind]
                    pca.partial_fit(wfs[:, :, chan_ind])
            else:
                Parallel(n_jobs=n_jobs)(delayed(partial_fit_one_channel)(pca_model_files[chan_ind], wfs[:, :, chan_ind])
                                                        for chan_ind in range(len(channel_ids)))

        # reload the models (if n_jobs > 1)
        if n_jobs not in (0, 1):
            pca_model = []
            for chan_ind, chan_id in enumerate(channel_ids):
                pca_model_file = pca_model_files[chan_ind]
                with open(pca_model_file, 'rb') as fid:
                    pca_model.append(pickle.load(fid))

        # save
        for chan_ind, chan_id in enumerate(channel_ids):
            pca = pca_model[chan_ind]
            with (self.extension_folder / f"pca_model_{mode}_{chan_id}.pkl").open("wb") as f:
                pickle.dump(pca, f)
            
        return pca_model

    def _run_by_channel_local(self, projection_memmap, n_jobs, progress_bar):
        """
        In this mode each PCA is "fit" and "transform" by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        pca_model = self._fit_by_channel_local(n_jobs, progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            if wfs.size == 0:
                continue
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = pca_model[chan_ind]
                proj = pca.transform(wfs[:, :, chan_ind])
                projection_memmap[unit_id][:, :, chan_ind] = proj

    def _fit_by_channel_global(self, progress_bar):
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        # there is one unique PCA accross channels
        pca_model = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        # with 'by_channel_global' we can't parallelize over channels
        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            if wfs.size == 0:
                continue
            # avoid loop with reshape
            shape = wfs.shape
            wfs_concat = wfs.transpose(0, 2, 1).reshape(shape[0] * shape[2], shape[1])
            pca_model.partial_fit(wfs_concat)

        # save
        mode = p["mode"]
        with (self.extension_folder / f"pca_model_{mode}.pkl").open("wb") as f:
            pickle.dump(pca_model, f)

        return pca_model

    def _run_by_channel_global(self, projection_memmap, n_jobs, progress_bar):
        """
        In this mode there is one "fit" for all channels.
        The transform is applied by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        pca_model = self._fit_by_channel_global(progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        # with 'by_channel_global' we can't parallelize over channels
        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            if wfs.size == 0:
                continue
            for chan_ind, chan_id in enumerate(channel_ids):
                proj = pca_model.transform(wfs[:, :, chan_ind])
                projection_memmap[unit_id][:, :, chan_ind] = proj
                
    def _fit_concatenated(self, progress_bar):
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids

        # there is one unique PCA accross channels
        pca_model = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Fitting PCA", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            pca_model.partial_fit(wfs_flat)

        # save
        mode = p["mode"]
        with (self.extension_folder / f"pca_model_{mode}.pkl").open("wb") as f:
            pickle.dump(pca_model, f)

        return pca_model

    def _run_concatenated(self, projection_memmap, n_jobs, progress_bar):
        """
        In this mode the waveforms are concatenated and there is
        a global fit_transform at once.
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids

        # there is one unique PCA accross channels
        pca_model = self._fit_concatenated(progress_bar)

        # transform
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="Projecting waveforms", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            wfs = we.get_waveforms(unit_id)
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            proj = pca_model.transform(wfs_flat)
            projection_memmap[unit_id][:, :] = proj


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
                                 n_components=5, mode='by_channel_local',
                                 whiten=True, dtype='float32', n_jobs=1,
                                 progress_bar=False):
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
    >>> we = si.extract_waveforms(recording, sorting, folder='waveforms_mearec')
    >>> pc = st.compute_principal_components(we, load_if_exists=True, n_components=3, mode='by_channel_local')
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

    folder = waveform_extractor.folder
    ext_folder = folder / WaveformPrincipalComponent.extension_name
    if load_if_exists and ext_folder.is_dir():
        pc = WaveformPrincipalComponent.load_from_folder(folder)
    else:
        pc = WaveformPrincipalComponent.create(waveform_extractor)
        pc.set_params(n_components=n_components, mode=mode, whiten=whiten, dtype=dtype)
        pc.run(n_jobs=n_jobs, progress_bar=progress_bar)

    return pc


def partial_fit_one_channel(pca_file, wf_chan):
    with open(pca_file, 'rb') as fid:
        pca_model = pickle.load(fid)
    pca_model.partial_fit(wf_chan)
    with pca_file.open("wb") as f:
        pickle.dump(pca_model, f)
