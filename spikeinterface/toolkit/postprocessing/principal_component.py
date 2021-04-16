import shutil
import json
from pathlib import Path

import numpy as np

from sklearn.decomposition import IncrementalPCA 

from spikeinterface.core.core_tools import check_json

from spikeinterface.core import WaveformExtractor


_possible_modes = [ 'by_channel_local', 'by_channel_global', 'concatenated']

class WaveformPrincipalComponent:
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
    
    def _run_by_channel_local(self, component_memmap):
        """
        In this mode each PCA is "fit" and "transform" by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params
        
        unit_ids = we.sorting.unit_ids
        channel_ids  = we.recording.channel_ids
        
        # there is one PCA per channel for independant fit per channel
        all_pca = [IncrementalPCA(n_components=p['n_components'], whiten=p['whiten']) for _ in channel_ids]
        
        # fit
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = all_pca[chan_ind]
                pca.partial_fit(wfs[:, :, chan_ind])
        
        # transform
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca = all_pca[chan_ind]
                comp = pca.transform(wfs[:, :, chan_ind])
                component_memmap[unit_id][:, :, chan_ind] = comp

    def _run_by_channel_global(self, component_memmap):
        """
        In this mode there is one "fit" for all channels.
        The transform is applied by channel.
        The output is then (n_spike, n_components, n_channels)
        """
        we = self.waveform_extractor
        p = self._params
        
        unit_ids = we.sorting.unit_ids
        channel_ids  = we.recording.channel_ids
        
        # there is one unique PCA accross channels
        pca = IncrementalPCA(n_components=p['n_components'], whiten=p['whiten'])
        
        # fit
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                pca.partial_fit(wfs[:, :, chan_ind])
        
        # transform
        for unit_id in unit_ids:
            wfs = we.get_waveforms(unit_id)
            for chan_ind, chan_id in enumerate(channel_ids):
                comp = pca.transform(wfs[:, :, chan_ind])
                component_memmap[unit_id][:, :, chan_ind] = comp


    def _run_concatenated(self, component_memmap):
        """
        In this mode the waveforms are concatenated and there is
        a global fit_stranfirom at once.
        """
        we = self.waveform_extractor
        p = self._params
        
        unit_ids = we.sorting.unit_ids
        channel_ids  = we.recording.channel_ids
        
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



def compute_principal_components(waveform_extractor, load_if_exists=False, 
            n_components=5, mode='by_channel_local',  whiten=True, dtype='float32'):
    """
    
    
    """
    
    folder = waveform_extractor.folder
    if load_if_exists and folder.is_dir() and (folder / 'PCA').is_dir():
        pc = WaveformPrincipalComponent.load_from_folder(folder)
    else:
        pc = WaveformPrincipalComponent.create(waveform_extractor)
        pc.set_params(n_components=n_components, mode=mode, whiten=whiten, dtype=dtype)
        print('ici')
        pc.run()
        print('ici')
    
    return pc

    
    
