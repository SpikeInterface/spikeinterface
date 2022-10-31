import pickle
from pathlib import Path
import shutil
import json

import numpy as np

from .base import load_extractor

from .core_tools import check_json
from .job_tools import _shared_job_kwargs_doc
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
import probeinterface
from .recording_tools import check_probe_do_not_overlap

_possible_template_modes = ('average', 'std', 'median')


class WaveformExtractor:
    """
    Class to extract waveform on paired Recording-Sorting objects.
    Waveforms are persistent on disk and cached in memory.

    Parameters
    ----------
    recording: Recording
        The recording object
    sorting: Sorting
        The sorting object
    folder: Path
        The folder where waveforms are cached
    rec_attributes: None or dict
        When recording is None then a minimal dict with some attributes
        is needed.
    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object

    Examples
    --------

    >>> # Instantiate
    >>> we = WaveformExtractor.create(recording, sorting, folder)

    >>> # Compute
    >>> we = we.set_params(...)
    >>> we = we.run_extract_waveforms(...)

    >>> # Retrieve
    >>> waveforms = we.get_waveforms(unit_id)
    >>> template = we.get_template(unit_id, mode='median')

    >>> # Load  from folder (in another session)
    >>> we = WaveformExtractor.load_from_folder(folder)

    """
    extensions = []
    def __init__(self, recording, sorting, folder=None, rec_attributes=None):
        if recording is None:
            # this is for the mode when recording is not accessible anymore
            if rec_attributes is None:
                raise ValueError('WaveformExtractor: if recording is None then rec_attributes must be provied')
            # some check on minimal attributes (probegroup is not mandatory)
            for k in ('channel_ids', 'sampling_frequency', 'num_channels'):
                if k not in rec_attributes:
                    raise ValueError(f'Missing key in rec_attributes {k}')
            self._rec_attributes = rec_attributes
        else:
            assert recording.get_num_segments() == sorting.get_num_segments(), \
                "The recording and sorting objects must have the same number of segments!"
            np.testing.assert_almost_equal(recording.get_sampling_frequency(),
                                           sorting.get_sampling_frequency(), decimal=2)
            if not recording.is_filtered():
                raise Exception('The recording is not filtered, you must filter it using `bandpass_filter()`.'
                                'If the recording is already filtered, you can also do '
                                '`recording.annotate(is_filtered=True)')

        self._recording = recording
        self.sorting = sorting

        # cache in memory
        self._waveforms = {}
        self._template_cache = {}
        self._params = {}
        self._loaded_extensions = dict()

        self.folder = folder
        if self.folder is not None:
            self.folder = Path(self.folder)
            if (self.folder / 'params.json').is_file():
                with open(str(self.folder / 'params.json'), 'r') as f:
                    self._params = json.load(f)
        else:
            # this is in case of in-memory
            self._memory_objects = {"wfs_arrays": {}, "sampled_indices": {}}

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        nunits = self.sorting.get_num_units()
        txt = f'{clsname}: {nchan} channels - {nunits} units - {nseg} segments'
        if len(self._params) > 0:
            max_spikes_per_unit = self._params['max_spikes_per_unit']
            txt = txt + f'\n  before:{self.nbefore} after:{self.nafter} n_per_units:{max_spikes_per_unit}'
        return txt

    @classmethod
    def load_from_folder(cls, folder, with_recording=True, sorting=None):
        folder = Path(folder)
        assert folder.is_dir(), f'This waveform folder does not exists {folder}'

        if not with_recording:
            # load
            recording = None
            rec_attributes_file = folder / 'recording_info' / 'recording_attributes.json'
            if not rec_attributes_file.exists():
                raise ValueError('This WaveformExtractor folder was created with an older version of spikeinterface'
                                 '\nYou cannot use the mode with_recording=False')
            with open(rec_attributes_file, 'r') as f:
                rec_attributes = json.load(f)
            # the probe is handle ouside the main json
            probegroup_file = folder / 'recording_info' / 'probegroup.json'
            if probegroup_file.is_file():
                rec_attributes['probegroup'] = probeinterface.read_probeinterface(probegroup_file)
            else:
                rec_attributes['probegroup'] = None
        else:
            try:
                recording = load_extractor(folder / 'recording.json',
                                        base_folder=folder)
                rec_attributes = None
            except:
                raise Exception("The recording could not be loaded. You can use the `with_recording=False` argument")

        if sorting is None:
            sorting = load_extractor(folder / 'sorting.json',
                                     base_folder=folder)
        we = cls(recording, sorting, folder=folder, rec_attributes=rec_attributes)

        for mode in _possible_template_modes:
            # load cached templates
            template_file = folder / f'templates_{mode}.npy'
            if template_file.is_file():
                we._template_cache[mode] = np.load(template_file)

        return we

    @classmethod
    def create(cls, recording, sorting, folder, mode="folder", remove_if_exists=False,
               use_relative_path=False):
        assert mode in ("folder", "memory")
        if mode == "folder":
            folder = Path(folder)
            if folder.is_dir():
                if remove_if_exists:
                    shutil.rmtree(folder)
                else:
                    raise FileExistsError('Folder already exists')
            folder.mkdir(parents=True)

            if use_relative_path:
                relative_to = folder
            else:
                relative_to = None

            if recording.is_dumpable:
                recording.dump(folder / 'recording.json', relative_to=relative_to)
            if sorting.is_dumpable:
                sorting.dump(folder / 'sorting.json', relative_to=relative_to)
            
            # dump some attributes of the recording for the mode with_recording=False at next load
            rec_attributes_file = folder / 'recording_info' / 'recording_attributes.json'
            rec_attributes_file.parent.mkdir()
            rec_attributes = dict(
                channel_ids=recording.channel_ids,
                sampling_frequency=recording.get_sampling_frequency(),
                num_channels=recording.get_num_channels(),
            )
            rec_attributes_file.write_text(
                json.dumps(check_json(rec_attributes), indent=4),
                encoding='utf8'
            )
            if recording.get_probegroup() is not None:
                probegroup_file = folder / 'recording_info' / 'probegroup.json'
                probeinterface.write_probeinterface(probegroup_file, recording.get_probegroup())

            with open(rec_attributes_file, 'r') as f:
                rec_attributes = json.load(f)

        return cls(recording, sorting, folder)

    @classmethod
    def register_extension(cls, extension_class):
        """
        This maintains a list of possible extensions that are available.
        It depends on the imported submodules (e.g. for postprocessing module).

        For instance:
        import spikeinterface as si
        si.WaveformExtractor.extensions == []

        from spikeinterface.postprocessing import WaveformPrincipalComponent
        si.WaveformExtractor.extensions == [WaveformPrincipalComponent, ...]

        """
        assert issubclass(extension_class, BaseWaveformExtractorExtension)
        assert extension_class.extension_name is not None, 'extension_name must not be None'
        assert all(extension_class.extension_name != ext.extension_name for ext in cls.extensions), \
            'Extension name already exists'
        cls.extensions.append(extension_class)
    
    # map some method from recording and sorting
    @property
    def recording(self):
        if self._recording is None:
            raise ValueError('WaveformExtractor is used in mode "with_recording=False" '
                             'this operation needs the recording')
        return self._recording

    @property
    def channel_ids(self):
        if self._recording is not None:
            return self.recording.channel_ids
        else:
            return np.array(self._rec_attributes['channel_ids'])
    
    @property
    def sampling_frequency(self):
        return self.sorting.get_sampling_frequency()

    @property
    def unit_ids(self):
        return self.sorting.unit_ids

    @property
    def nbefore(self):
        nbefore = int(self._params['ms_before'] * self.sampling_frequency / 1000.)
        return nbefore

    @property
    def nafter(self):
        nafter = int(self._params['ms_after'] * self.sampling_frequency / 1000.)
        return nafter

    @property
    def nsamples(self):
        return self.nbefore + self.nafter

    @property
    def return_scaled(self):
        return self._params['return_scaled']
     
    def get_num_channels(self):
        if self._recording is not None:
            return self.recording.get_num_channels()
        else:
            return self._rec_attributes['num_channels']

    def get_num_segments(self):
        return self.sorting.get_num_segments()

    def get_probegroup(self):
        if self._recording is not None:
            return self.recording.get_probegroup()
        else:
            return self._rec_attributes['probegroup']
    
    def get_probe(self):
        probegroup = self.get_probegroup()
        assert len(probegroup.probes) == 1, 'There are several probes. Use `get_probegroup()`'
        return probegroup.probes[0]

    def get_channel_locations(self):
        # important note : contrary to recording
        # this give all channel locations, so no kwargs like channel_ids and axes
        if self._recording is not None:
            return self.recording.get_channel_locations()
        else:
            if self.get_probegroup() is not None:
                all_probes = self.get_probegroup().probes
                # check that multiple probes are non-overlapping
                check_probe_do_not_overlap(all_probes)
                all_positions = np.vstack([probe.contact_positions for probe in all_probes])
                return all_positions
            else:
                raise Exception('There are no channel locations')

    def channel_ids_to_indices(self, channel_ids):
        if self._recording is not None:
            return self.recording.ids_to_indices(channel_ids)
        else:
            all_channel_ids = self._rec_attributes['channel_ids']
            indices = np.array([all_channel_ids.index(id) for id in channel_ids], dtype=int)
            return indices

    def get_extension_class(self, extension_name):
        """
        Get extension class from name and check if registered.

        Parameters
        ----------
        extension_name: str
            The extension name.

        Returns
        -------
        ext_class:
            The class of the extension.
        """
        extensions_dict = {ext.extension_name: ext for ext in self.extensions}
        assert extension_name in extensions_dict, \
            'Extension is not registered, please import related module before'
        ext_class = extensions_dict[extension_name]
        return ext_class

    def is_extension(self, extension_name):
        """
        Check if the extension exists in memory or in the folder.

        Parameters
        ----------
        extension_name: str
            The extension name.

        Returns
        -------
        exists: bool
            Whether the extension exists or not
        """
        if self.folder is None:
            return extension_name in self._loaded_extensions
        else:
            return (self.folder / extension_name).is_dir() and (self.folder / extension_name / 'params.json').is_file()

    def load_extension(self, extension_name):
        """
        Load an extension from its name.
        The module of the extension must be loaded and registered.

        Parameters
        ----------
        extension_name: str
            The extension name.

        Returns
        -------
        ext_instanace: 
            The loaded instance of the extension
        """
        if self.folder is not None and extension_name not in self._loaded_extensions:
            if self.is_extension(extension_name):
                ext_class = self.get_extension_class(extension_name)
                ext = ext_class.load_from_folder(self.folder, self)
        if extension_name not in self._loaded_extensions:
            raise Exception(f'Extension {extension_name} not available')
        return self._loaded_extensions[extension_name]

    def delete_extension(self, extension_name):
        """
        Deletes an existing extension.

        Parameters
        ----------
        extension_name: str
            The extension name.
        """
        assert self.is_extension(extension_name), f"The extension {extension_name} is not available"
        del self._loaded_extensions[extension_name]
        if self.folder is not None and (self.folder / extension_name).is_dir():
            shutil.rmtree(self.folder / extension_name)

    def get_available_extension_names(self):
        """
        Return a list of loaded or available extension names either in memory or
        in persistent extension folders.
        Then instances can be loaded with we.load_extension(extension_name)

        Importante note: extension modules need to be loaded (and so registered)
        before this call, otherwise extensions will be ignored even if the folder
        exists.

        Returns
        -------
        extension_names_in_folder: list
            A list of names of computed extension in this folder
        """
        extension_names_in_folder = []
        for extension_class in self.extensions:
            if self.is_extension(extension_class.extension_name):
                extension_names_in_folder.append(extension_class.extension_name)
        return extension_names_in_folder

    def _reset(self):
        self._waveforms = {}
        self._template_cache = {}
        self._params = {}

        if self.folder is not None:
            waveform_folder = self.folder / 'waveforms'
            if waveform_folder.is_dir():
                shutil.rmtree(waveform_folder)
            for mode in _possible_template_modes:
                template_file = self.folder / f'templates_{mode}.npy'
                if template_file.is_file():
                    template_file.unlink()

            waveform_folder.mkdir()
        else:
            # remove shared objects
            self._memory_objects = {"wfs_arrays": {}, "sampled_indices": {}}

    def set_params(self, ms_before=1., ms_after=2., max_spikes_per_unit=500, return_scaled=False, dtype=None):
        """
        Set parameters for waveform extraction

        Parameters
        ----------
        ms_before: float
            Cut out in ms before spike time
        ms_after: float
            Cut out in ms after spike time
        max_spikes_per_unit: int
            Maximum number of spikes to extract per unit
        return_scaled: bool
            If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV.
        dtype: np.dtype
            The dtype of the computed waveforms
        """
        self._reset()

        if dtype is None:
            dtype = self.recording.get_dtype()

        if return_scaled:
            # check if has scaled values:
            if not self.recording.has_scaled():
                print("Setting 'return_scaled' to False")
                return_scaled = False

        if np.issubdtype(dtype, np.integer) and return_scaled:
            dtype = "float32"

        dtype = np.dtype(dtype)

        if max_spikes_per_unit is not None:
            max_spikes_per_unit = int(max_spikes_per_unit)

        self._params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=max_spikes_per_unit,
            return_scaled=return_scaled,
            dtype=dtype.str)

        if self.folder is not None:
            (self.folder / 'params.json').write_text(
                json.dumps(check_json(self._params), indent=4), encoding='utf8')
        
    def select_units(self, unit_ids, new_folder=None, use_relative_path=False):
        """
        Filters units by creating a new waveform extractor object in a new folder.
        
        Extensions are also updated to filter the selected unit ids.

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new WaveformExtractor object
        new_folder : Path or None
            The new folder where selected waveforms are copied
            
        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        sorting = self.sorting.select_units(unit_ids)
        unit_indices = self.sorting.ids_to_indices(unit_ids)

        if self.folder is not None:
            assert new_folder is not None, "Please specify 'new_folder'"
            new_folder = Path(new_folder)
            assert not new_folder.is_dir(), f"{new_folder} already exists!"
            new_folder.mkdir(parents=True)

            # create new waveform extractor folder
            shutil.copyfile(self.folder / "params.json",
                            new_folder / "params.json")
            shutil.copyfile(self.folder / "recording.json",
                            new_folder / "recording.json")

            if use_relative_path:
                relative_to = new_folder
            else:
                relative_to = None

            sorting.dump(new_folder / 'sorting.json', relative_to=relative_to)

            # create and populate waveforms folder
            new_waveforms_folder = new_folder / "waveforms"
            new_waveforms_folder.mkdir()
        
            waveforms_files = [f for f in (self.folder / "waveforms").iterdir() if f.suffix == ".npy"]
            for unit in sorting.get_unit_ids():
                for wf_file in waveforms_files:
                    if f"waveforms_{unit}.npy" in wf_file.name or f'sampled_index_{unit}.npy' in wf_file.name:
                        shutil.copyfile(
                            wf_file, new_waveforms_folder / wf_file.name)

            template_files = [f for f in self.folder.iterdir() if "template" in f.name and f.suffix == ".npy"]
            for tmp_file in template_files:
                templates_data_sliced = np.load(tmp_file)[unit_indices]
                np.save(new_waveforms_folder / tmp_file.name, templates_data_sliced)
            we = WaveformExtractor.load_from_folder(new_folder)
        else:
            sorting = self.sorting.select_units(unit_ids)
            we = WaveformExtractor.create(self.recording, sorting, folder=None, mode="memory")
            we.set_params(**self._params)
            # copy memory objects
            for unit_id in unit_ids:
                we._memory_objects["wfs_arrays"][unit_id] = self._memory_objects["wfs_arrays"][unit_id]
                we._memory_objects["sampled_indices"][unit_id] = self._memory_objects["sampled_indices"][unit_id]

        # finally select extensions data
        for ext_name in self.get_available_extension_names():
            ext = self.load_extension(ext_name)
            ext.select_units(unit_ids, new_waveform_extractor=we)

        return we

    def get_waveforms(self, unit_id, with_index=False, cache=False, memmap=True, sparsity=None):
        """
        Return waveforms for the specified unit id.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)
        cache: bool
            If True, waveforms are cached to the self._waveforms dictionary (default False)
        memmap: bool
            If True, waveforms are loaded as memmap objects.
            If False, waveforms are loaded as np.array objects (default True)
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity by channel ids as values.
            The sparsity can be computed with the get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
        indices: np.array
            If 'with_index' is True, the spike indices corresponding to the waveforms extracted
        """
        assert unit_id in self.sorting.unit_ids, "'unit_id' is invalid"

        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            if self.folder is not None:
                waveform_file = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
                if not waveform_file.is_file():
                    raise Exception('Waveforms not extracted yet: '
                                    'please do WaveformExtractor.run_extract_waveforms() first')
                if memmap:
                    wfs = np.load(str(waveform_file), mmap_mode="r")
                else:
                    wfs = np.load(waveform_file)
                if cache:
                    self._waveforms[unit_id] = wfs
            else:
                wfs = self._memory_objects["wfs_arrays"][unit_id]

        if sparsity is not None:
            assert unit_id in sparsity, f"Sparsity for unit {unit_id} is not in the sparsity dictionary!"
            chan_inds = self.recording.ids_to_indices(sparsity[unit_id])
            wfs = wfs[:, :, chan_inds]

        if with_index:
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def get_sampled_indices(self, unit_id):
        """
        Return sampled spike indices of extracted waveforms

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve indices for

        Returns
        -------
        sampled_indices: np.array
            The sampled indices
        """
        if self.folder is not None:
            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            sampled_index = np.load(sampled_index_file)
        else:
            sampled_index = self._memory_objects["sampled_indices"][unit_id]
        return sampled_index

    def get_waveforms_segment(self, segment_index, unit_id, sparsity=None):
        """
        Return waveforms from a specified segment and unit_id.

        Parameters
        ----------
        segment_index: int
            The segment index to retrieve waveforms from
        unit_id: int or str
            Unit id to retrieve waveforms for
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity by index as values.
            The sparsity can be computed with the get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
        """
        wfs, index_ar = self.get_waveforms(unit_id, with_index=True, sparsity=sparsity)
        mask = index_ar['segment_index'] == segment_index
        return wfs[mask, :, :]

    def precompute_templates(self, modes=('average', 'std')):
        """
        Precompute all template for different "modes":
          * average
          * std
          * median

        The results is cache in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.
        """
        # TODO : run this in parralel

        unit_ids = self.unit_ids
        num_chans = self.get_num_channels()

        for mode in modes:
            dtype = self._params['dtype'] if mode == 'median' else np.float32
            templates = np.zeros((len(unit_ids), self.nsamples, num_chans), dtype=dtype)
            self._template_cache[mode] = templates

        for i, unit_id in enumerate(unit_ids):
            wfs = self.get_waveforms(unit_id, cache=False)
            for mode in modes:
                if len(wfs) == 0:
                    arr = np.zeros(wfs.shape[1:], dtype=wfs.dtype)
                elif mode == 'median':
                    arr = np.median(wfs, axis=0)
                elif mode == 'average':
                    arr = np.average(wfs, axis=0)
                elif mode == 'std':
                    arr = np.std(wfs, axis=0)
                else:
                    raise ValueError('mode must in median/average/std')

                self._template_cache[mode][i, :, :] = arr

        for mode in modes:
            templates = self._template_cache[mode]
            if self.folder is not None:
                template_file = self.folder / f'templates_{mode}.npy'
                np.save(template_file, templates)

    def get_all_templates(self, unit_ids=None, mode='average'):
        """
        Return  templates (average waveform) for multiple units.

        Parameters
        ----------
        unit_ids: list or None
            Unit ids to retrieve waveforms for
        mode: str
            'average' (default) or 'median' , 'std'

        Returns
        -------
        templates: np.array
            The returned templates (num_units, num_samples, num_channels)
        """
        if mode not in self._template_cache:
            self.precompute_templates(modes=[mode])

        templates = self._template_cache[mode]

        if unit_ids is not None:
            unit_indices = self.sorting.ids_to_indices(unit_ids)
            templates = templates[unit_indices, :, :]

        return templates

    def get_template(self, unit_id, mode='average', sparsity=None):
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        mode: str
            'average' (default), 'median' , 'std'(standard deviation)
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity as values.
            The sparsity can be computed with the get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """
        assert mode in _possible_template_modes
        assert unit_id in self.sorting.unit_ids

        key = mode

        if mode in self._template_cache:
            # already in the global cache
            templates = self._template_cache[mode]
            unit_ind = self.sorting.id_to_index(unit_id)
            template = templates[unit_ind, :, :]
            if sparsity is not None:
                chan_inds = self.recording.ids_to_indices(sparsity[unit_id])
                template = template[:, chan_inds]
            return template

        # compute from waveforms
        wfs = self.get_waveforms(unit_id, sparsity=sparsity)
        if mode == 'median':
            template = np.median(wfs, axis=0)
        elif mode == 'average':
            template = np.average(wfs, axis=0)
        elif mode == 'std':
            template = np.std(wfs, axis=0)
        return template

    def get_template_segment(self, unit_id, segment_index, mode='average',
                             sparsity=None):
        """
        Return template for the specified unit id computed from waveforms of a specific segment.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        segment_index: int
            The segment index to retrieve template from
        mode: str
            'average'  (default), 'median', 'std'(standard deviation)
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity by index as values.
            The sparsity can be computed with the get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)

        """
        assert mode in ('median', 'average', 'std', )
        assert unit_id in self.sorting.unit_ids
        waveforms_segment = self.get_waveforms_segment(segment_index, unit_id,
                                                       sparsity=sparsity)
        if mode == 'median':
            return np.median(waveforms_segment, axis=0)
        elif mode == 'average':
            return np.mean(waveforms_segment, axis=0)
        elif mode == 'std':
            return np.std(waveforms_segment, axis=0)

    def sample_spikes(self, seed=None):
        nbefore = self.nbefore
        nafter = self.nafter

        selected_spikes = select_random_spikes_uniformly(self.recording, self.sorting,
                                                         self._params['max_spikes_per_unit'], 
                                                         nbefore, nafter, seed)

        # store in a 2 columns (spike_index, segment_index) in a npy file
        for unit_id in self.sorting.unit_ids:
            n = np.sum([e.size for e in selected_spikes[unit_id]])
            sampled_index = np.zeros(n, dtype=[('spike_index', 'int64'), ('segment_index', 'int64')])
            pos = 0
            for segment_index in range(self.sorting.get_num_segments()):
                inds = selected_spikes[unit_id][segment_index]
                sampled_index[pos:pos + inds.size]['spike_index'] = inds
                sampled_index[pos:pos + inds.size]['segment_index'] = segment_index
                pos += inds.size

            if self.folder is not None:
                sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
                np.save(sampled_index_file, sampled_index)
            else:
                self._memory_objects["sampled_indices"][unit_id] = sampled_index

        return selected_spikes

    def run_extract_waveforms(self, seed=None, **job_kwargs):
        p = self._params
        nbefore = self.nbefore
        nafter = self.nafter
        return_scaled = self.return_scaled
        unit_ids = self.sorting.unit_ids

        selected_spikes = self.sample_spikes(seed=seed)

        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                selected_spike_times[unit_id].append(spike_times[sel])
        
        spikes = []
        for segment_index in range(self.sorting.get_num_segments()):
            num_in_seg = np.sum([selected_spikes[unit_id][segment_index].size for unit_id in unit_ids])
            spike_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
            spikes_ = np.zeros(num_in_seg,  dtype=spike_dtype)
            pos = 0
            for unit_ind, unit_id in enumerate(unit_ids):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                n = sel.size
                spikes_[pos:pos+n]['sample_ind'] = spike_times[sel]
                spikes_[pos:pos+n]['unit_ind'] = unit_ind
                spikes_[pos:pos+n]['segment_ind'] = segment_index
                pos += n
            order = np.argsort(spikes_)
            spikes_ = spikes_[order]
            spikes.append(spikes_)
        spikes = np.concatenate(spikes)

        if self.folder is not None:
            wf_folder = self.folder / 'waveforms'
            mode = "memmap"
            copy = False
        else:
            wf_folder = None
            mode = "shared_memory"
            copy = True
        wfs_arrays = extract_waveforms_to_buffers(self.recording, spikes, unit_ids, nbefore, nafter,
                                                  mode=mode, return_scaled=return_scaled, folder=wf_folder,
                                                  dtype=p['dtype'], sparsity_mask=None, copy=copy, **job_kwargs)
        if self.folder is None:
            self._memory_objects["wfs_arrays"] = wfs_arrays


def select_random_spikes_uniformly(recording, sorting, max_spikes_per_unit, nbefore=None, nafter=None, seed=None):
    """
    Uniform random selection of spike across segment per units.

    This function does not select spikes near border if nbefore/nafter are not None.
    """
    unit_ids = sorting.unit_ids
    num_seg = sorting.get_num_segments()

    if seed is not None:
        np.random.seed(int(seed))

    selected_spikes = {}
    for unit_id in unit_ids:
        # spike per segment
        n_per_segment = [sorting.get_unit_spike_train(unit_id, segment_index=i).size for i in range(num_seg)]
        cum_sum = [0] + np.cumsum(n_per_segment).tolist()
        total = np.sum(n_per_segment)
        if max_spikes_per_unit is not None:
            if total > max_spikes_per_unit:
                global_inds = np.random.choice(total, size=max_spikes_per_unit, replace=False)
                global_inds = np.sort(global_inds)
            else:
                global_inds = np.arange(total)
        else:
            global_inds = np.arange(total)
        sel_spikes = []
        for segment_index in range(num_seg):
            in_segment = (global_inds >= cum_sum[segment_index]) & (global_inds < cum_sum[segment_index + 1])
            inds = global_inds[in_segment] - cum_sum[segment_index]

            if max_spikes_per_unit is not None:
                # clean border when sub selection
                assert nafter is not None
                spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sampled_spike_times = spike_times[inds]
                num_samples = recording.get_num_samples(segment_index=segment_index)
                mask = (sampled_spike_times >= nbefore) & (sampled_spike_times < (num_samples - nafter))
                inds = inds[mask]

            sel_spikes.append(inds)
        selected_spikes[unit_id] = sel_spikes
    return selected_spikes


def extract_waveforms(recording, sorting, folder=None,
                      mode='folder', load_if_exists=False,
                      precompute_template=('average', ),
                      ms_before=3., ms_after=4.,
                      max_spikes_per_unit=500,
                      overwrite=False,
                      return_scaled=True,
                      dtype=None,
                      use_relative_path=False,
                      seed=None,
                      **job_kwargs):
    """
    Extracts waveform on paired Recording-Sorting objects.
    Waveforms are persistent on disk and cached in memory.

    Parameters
    ----------
    recording: Recording
        The recording object
    sorting: Sorting
        The sorting object
    folder: str or Path or None
        The folder where waveforms are cached
    mode: str
        "folder" (default) or "memory". The "folder" argument must be specified in case of mode "folder".
        If "memory" is used, the waveforms are stored in RAM. Use this option carefully!
    load_if_exists: bool
        If True and waveforms have already been extracted in the specified folder, they are loaded
        and not recomputed.
    precompute_template: None or list
        Precompute average/std/median for template. If None not precompute.
    ms_before: float
        Time in ms to cut before spike peak
    ms_after: float
        Time in ms to cut after spike peak
    max_spikes_per_unit: int or None
        Number of spikes per unit to extract waveforms from (default 500).
        Use None to extract waveforms for all spikes
    overwrite: bool
        If True and 'folder' exists, the folder is removed and waveforms are recomputed.
        Otherwise an error is raised.
    return_scaled: bool
        If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV.
    dtype: dtype or None
        Dtype of the output waveforms. If None, the recording dtype is maintained.
    use_relative_path: bool
        If True, the recording and sorting paths are relative to the waveforms folder. 
        This allows portability of the waveform folder provided that the relative paths are the same, 
        but forces all the data files to be in the same drive.
        Default is False.
    seed: int or None
        Random seed for spike selection

    {}

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object

    """
    if mode == "folder":
        assert folder is not None
        folder = Path(folder)
        assert not (overwrite and load_if_exists), "Use either 'overwrite=True' or 'load_if_exists=True'"
        if overwrite and folder.is_dir():
            shutil.rmtree(folder)
        if load_if_exists and folder.is_dir():
            we = WaveformExtractor.load_from_folder(folder)
            return we
    we = WaveformExtractor.create(recording, sorting, folder, mode=mode, use_relative_path=use_relative_path)
    we.set_params(ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=max_spikes_per_unit, dtype=dtype,
                  return_scaled=return_scaled)
    we.run_extract_waveforms(seed=seed, **job_kwargs)

    if precompute_template is not None:
        we.precompute_templates(modes=precompute_template)

    return we


extract_waveforms.__doc__ = extract_waveforms.__doc__.format(_shared_job_kwargs_doc)



class BaseWaveformExtractorExtension:
    """
    This the base class to extend the waveform extractor.
    It handles persistency to disk any computations related
    to a waveform extractor.
    
    For instance:
      * principal components
      * spike amplitudes
      * quality metrics

    The design is done via a `WaveformExtractor.register_extension(my_extension_class)`,
    so that only imported modules can be used as *extension*.

    It also enables any custum computation on top on waveform extractor to be implemented by the user.
    
    An extension needs to inherit from this class and implement some abstract methods:
      * _reset
      * _set_params
      * _run
    
    The subclass must also save to the `self.extension_folder` any file that needs
    to be reloaded when calling `_specific_load_from_folder`

    The subclass must also set an `extension_name` attribute which is not None by default.
    """
    
    # must be set in inherited in subclass 
    extension_name = None
    
    def __init__(self, waveform_extractor):
        self.waveform_extractor = waveform_extractor
        
        if self.waveform_extractor.folder is not None:
            self.folder = self.waveform_extractor.folder
            self.extension_folder = self.folder / self.extension_name

            if not self.extension_folder.is_dir():
                self.extension_folder.mkdir()
        else:
            self.extension_folder = None
            self.folder = None
        self._extension_data = dict()
        self._params = None

        # register
        self.waveform_extractor._loaded_extensions[self.extension_name] = self

    @classmethod
    def load_from_folder(cls, folder, waveform_extractor=None):
        """
        Load extension from folder.
        'folder' is the waveform extractor folder.
        """
        ext_folder = Path(folder) / cls.extension_name
        assert ext_folder.is_dir(), f'WaveformExtractor: extension {cls.extension_name} is not in folder {folder}'
        
        params_file = ext_folder / 'params.json'
        assert params_file.is_file(), f'No params file in extension {cls.extension_name} folder'
        
        with open(str(params_file), 'r') as f:
            params = json.load(f)

        if waveform_extractor is None:
            waveform_extractor = WaveformExtractor.load_from_folder(folder)
        
        # make instance with params
        ext = cls(waveform_extractor)
        ext._params = params
        ext._load_extension_data()

        return ext

    # use load instead
    def _load_extension_data(self):
        for ext_data_file in self.extension_folder.iterdir():
            if ext_data_file.name == 'params.json':
                continue
            ext_data_name = ext_data_file.stem
            if ext_data_file.suffix == '.json':
                data = json.load(ext_data_file.open('r'))
            elif ext_data_file.suffix == '.npy':
                data = np.load(ext_data_file, mmap_mode='r')
            elif ext_data_file.suffix == '.csv':
                import pandas as pd
                data = pd.read_csv(ext_data_file, index_col=0)
            elif ext_data_file.suffix == '.pkl':
                data = pickle.load(ext_data_file.open('rb'))
            self._extension_data[ext_data_name] = data

    def run(self, **kwargs):
        self._run(**kwargs)
        self._save()

    def _run(self, **kwargs):
        # must be implemented in subclass
        # must populate the self._extension_data dictionary
        raise NotImplementedError

    def _save(self):
        if self.folder is not None:
            import pandas as pd
            for ext_data_name, ext_data in self._extension_data.items():
                if isinstance(ext_data, dict):
                    with (self.extension_folder / f"{ext_data_name}.json").open('w') as f:
                        json.dump(ext_data, f)
                elif isinstance(ext_data, np.ndarray):
                    np.save(self.extension_folder / f"{ext_data_name}.npy", ext_data)
                elif isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(self.extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (self.extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")

    def reset(self):
        """
        Reset the waveform extension.
        Delete the sub folder and create a new empty one.
        """        
        if self.extension_folder is not None:
            if self.extension_folder.is_dir():
                shutil.rmtree(self.extension_folder)
            self.extension_folder.mkdir()
        
        self._params = None
        self._extension_data = dict()
    
    def select_units(self, unit_ids, new_waveform_extractor):
        new_extension = self.__class__(new_waveform_extractor)
        new_extension.set_params(**self._params)
        new_extension_data = self._select_extension_data(unit_ids=unit_ids)
        new_extension._extension_data = new_extension_data
        new_extension._save()
        
    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

    def set_params(self, **params):
        """
        Set parameters for the extension and
        make it persistent in json.
        """
        params = self._set_params(**params)
        self._params = params
        if self.extension_folder is not None:
            param_file = self.extension_folder / 'params.json'
            param_file.write_text(json.dumps(check_json(self._params), indent=4), encoding='utf8')
    
    def _set_params(self, **params):
        # must be implemented in subclass
        # must return a cleaned version of params dict
        raise NotImplementedError

    @staticmethod
    def get_extension_function():
        # must be implemented in subclass
        # must return extension function
        raise NotImplementedError

