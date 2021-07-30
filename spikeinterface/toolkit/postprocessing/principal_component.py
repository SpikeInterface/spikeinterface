import shutil
import json
from pathlib import Path

import numpy as np

from sklearn.decomposition import IncrementalPCA

from spikeinterface.core.core_tools import check_json
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs
from spikeinterface.core import WaveformExtractor
from .template_tools import get_template_channel_sparsity

_possible_modes = ['by_channel_local', 'by_channel_global', 'concatenated']


class WaveformPrincipalComponent:
    """
    Class to extract principal components from a WaveformExtractor object.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The WaveformExtractor object

    Returns
    -------
    pc: WaveformPrincipalComponent
        The WaveformPrincipalComponent object

    Examples
    --------
    >>> we = si.extract_waveforms(recording, sorting, folder='waveforms_mearec')
    >>> pc = st.compute_principal_components(we, load_if_exists=True, n_components=3, mode='by_channel_local')
    >>> components = pc.get_components(unit_id=1)
    >>> all_components = pc.get_all_components()

    """

    def __init__(self, waveform_extractor):
        self.waveform_extractor = waveform_extractor

        self.folder = self.waveform_extractor.folder

        self._params = {}
        if (self.folder / 'params_pca.json').is_file():
            with open(str(self.folder / 'params_pca.json'), 'r') as f:
                self._params = json.load(f)

    @classmethod
    def load_from_folder(cls, folder):
        we = WaveformExtractor.load_from_folder(folder)
        pc = WaveformPrincipalComponent(we)
        return pc

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
        self._components = {}
        self._params = {}

        pca_folder = self.folder / 'PCA'
        if pca_folder.is_dir():
            shutil.rmtree(pca_folder)
        pca_folder.mkdir()

    def set_params(self, n_components=5, mode='by_channel_local',
                   whiten=True, dtype='float32'):
        """
        Set parameters for waveform extraction

        Parameters
        ----------
        n_components:  int
        
        mode : 'by_channel_local' / 'by_channel_global' / 'concatenated'
        
        whiten: bool
            params transmitted to sklearn.PCA
        
        """
        self._reset()

        assert mode in _possible_modes

        self._params = dict(
            n_components=int(n_components),
            mode=str(mode),
            whiten=bool(whiten),
            dtype=np.dtype(dtype).str)

        (self.folder / 'params_pca.json').write_text(
            json.dumps(check_json(self._params), indent=4), encoding='utf8')

    def get_components(self, unit_id):
        component_file = self.folder / 'PCA' / f'pca_{unit_id}.npy'
        comp = np.load(component_file)
        return comp

    def get_all_components(self, channel_ids=None, unit_ids=None):
        recording = self.waveform_extractor.recording

        if unit_ids is None:
            unit_ids = self.waveform_extractor.sorting.unit_ids

        all_labels = []
        all_components = []
        for unit_id in unit_ids:
            comp = self.get_components(unit_id)
            if channel_ids is not None:
                chan_inds = recording.ids_to_indices(channel_ids)
                comp = comp[:, :, chan_inds]
            n = comp.shape[0]
            labels = np.array([unit_id] * n)
            all_labels.append(labels)
            all_components.append(comp)
        all_labels = np.concatenate(all_labels, axis=0)
        all_components = np.concatenate(all_components, axis=0)

        return all_labels, all_components

    def run(self):
        """
        This compute the PCs on waveforms extacted within
        the WaveformExtarctor.
        It is only for some sampled spikes defined in WaveformExtarctor
        
        The index of spikes come from the WaveformExtarctor.
        This will be cached in the same folder than WaveformExtarctor
        in 'PCA' subfolder.
        """
        p = self._params
        we = self.waveform_extractor
        num_chans = we.recording.get_num_channels()

        # prepare memmap files with npy
        component_memmap = {}
        unit_ids = we.sorting.unit_ids
        for unit_id in unit_ids:
            n_spike = we.get_waveforms(unit_id).shape[0]
            component_file = self.folder / 'PCA' / f'pca_{unit_id}.npy'
            if p['mode'] in ('by_channel_local', 'by_channel_global'):
                shape = (n_spike, p['n_components'], num_chans)
            elif p['mode'] == 'concatenated':
                shape = (n_spike, p['n_components'])
            comp = np.zeros(shape, dtype=p['dtype'])
            np.save(component_file, comp)
            comp = np.load(component_file, mmap_mode='r+')
            component_memmap[unit_id] = comp

        # run ...
        if p['mode'] == 'by_channel_local':
            self._run_by_channel_local(component_memmap)
        elif p['mode'] == 'by_channel_global':
            self._run_by_channel_local(component_memmap)
        elif p['mode'] == 'concatenated':
            self._run_concatenated(component_memmap)

    def run_for_all_spikes(self, file_path, max_channels_per_template=16, peak_sign='neg',
                           **job_kwargs):
        """
        This run the PCs on all spikes from the sorting.
        This is a long computation because waveform need to be extracted from each spikes.
        
        Used mainly for `export_to_phy()`
        
        PCs are exported to a .npy single file.
        
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

        best_channels_index = get_template_channel_sparsity(we, method='best_channels',
                                                            peak_sign=peak_sign, num_channels=max_channels_per_template,
                                                            outputs='index')

        unit_channels = [best_channels_index[unit_id] for unit_id in sorting.unit_ids]

        if p['mode'] == 'by_channel_local':
            all_pca = self._fit_by_channel_local()
        elif p['mode'] == 'by_channel_global':
            one_pca = self._fit_by_channel_global()
            all_pca = [one_pca] * recording.get_num_channels()

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
        init_args = init_args + (all_pcs_args, spike_times, spike_labels, we.nbefore, we.nafter, unit_channels, all_pca)
        processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name='extract PCs', **job_kwargs)
        processor.run()

    def _fit_by_channel_local(self):
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        # there is one PCA per channel for independent fit per channel
        all_pca = [IncrementalPCA(n_components=p['n_components'], whiten=p['whiten']) for _ in channel_ids]

        # fit
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = all_pca[chan_ind]
                pca.partial_fit(wfs[:, :, chan_ind])

        return all_pca

    def _run_by_channel_local(self, component_memmap):
        """
        In this mode each PCA is "fit" and "transform" by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        all_pca = self._fit_by_channel_local()

        # transform
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = all_pca[chan_ind]
                comp = pca.transform(wfs[:, :, chan_ind])
                component_memmap[unit_id][:, :, chan_ind] = comp

    def _fit_by_channel_global(self):
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        # there is one unique PCA accross channels
        one_pca = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                one_pca.partial_fit(wfs[:, :, chan_ind])

        return one_pca

    def _run_by_channel_global(self, component_memmap):
        """
        In this mode there is one "fit" for all channels.
        The transform is applied by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        one_pca = self._fit_by_channel_global()

        # transform
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                comp = one_pca.transform(wfs[:, :, chan_ind])
                component_memmap[unit_id][:, :, chan_ind] = comp

    def _run_concatenated(self, component_memmap):
        """
        In this mode the waveforms are concatenated and there is
        a global fit_stranfirom at once.
        """
        we = self.waveform_extractor
        p = self._params

        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids

        # there is one unique PCA accross channels
        pca = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])

        # fit
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            wfs_fat = wfs.reshape(wfs.shape[0], -1)
            pca.partial_fit(wfs_fat)

        # transform
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            wfs_fat = wfs.reshape(wfs.shape[0], -1)
            comp = pca.transform(wfs_fat)
            component_memmap[unit_id][:, :] = comp


def _all_pc_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx['recording']
    all_pcs = worker_ctx['all_pcs']
    spike_times = worker_ctx['spike_times']
    spike_labels = worker_ctx['spike_labels']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    unit_channels = worker_ctx['unit_channels']
    all_pca = worker_ctx['all_pca']

    i0 = np.searchsorted(spike_times, start_frame)
    i1 = np.searchsorted(spike_times, end_frame)

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
                all_pcs[i, :, c] = all_pca[chan_ind].transform(w)


def _init_work_all_pc_extractor(recording, all_pcs_args, spike_times, spike_labels, nbefore, nafter, unit_channels,
                                all_pca):
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
    worker_ctx['all_pca'] = all_pca

    return worker_ctx


def compute_principal_components(waveform_extractor, load_if_exists=False,
                                 n_components=5, mode='by_channel_local',
                                 whiten=True, dtype='float32'):
    """
    Compute PC scores from waveform extractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    load_if_exists: bool
        If True and pc scores are already in the waveform extractor folders, pc scores are loaded and not recomputed.
    n_components: int
        Number of components fo PCA
    mode: str
        - 'by_channel_local': a local PCA is fitted for each channel (projection by channel)
        - 'by_channel_global': a global PCA is fitted for all channels (projection by channel)
        - 'concatenated': channels are concatenated and a global PCA is fitted
    whiten: bool
        If True, waveforms are pre-whitened
    dtype: dtype
        Dtype of the pc scores (default float32)

    Returns
    -------
    pc: WaveformPrincipalComponent
        The waveform principal component object
    """

    folder = waveform_extractor.folder
    if load_if_exists and folder.is_dir() and (folder / 'PCA').is_dir():
        pc = WaveformPrincipalComponent.load_from_folder(folder)
    else:
        pc = WaveformPrincipalComponent.create(waveform_extractor)
        pc.set_params(n_components=n_components, mode=mode, whiten=whiten, dtype=dtype)
        pc.run()

    return pc
