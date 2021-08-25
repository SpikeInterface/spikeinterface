from pathlib import Path
import shutil
import json

import numpy as np

from .base import load_extractor

from .core_tools import check_json
from .job_tools import ChunkRecordingExecutor, ensure_n_jobs, _shared_job_kwargs_doc


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
    >>> we = we.run(...)
    
    >>> # Retrieve
    >>> waveforms = we.get_waveforms(unit_id)
    >>> template = we.get_template(unit_id, mode='median')
    
    >>> # Load  from folder (in another session)
    >>> we = WaveformExtractor.load_from_folder(folder)
    
    """

    def __init__(self, recording, sorting, folder):
        assert recording.get_num_segments() == sorting.get_num_segments(), \
            "The recording and sorting objects must have the same number of segments!"
        np.testing.assert_almost_equal(recording.get_sampling_frequency(),
                                       sorting.get_sampling_frequency(), decimal=2)

        if not recording.is_filtered():
            raise Exception('The recording is not filtered, you must filter it using `bandpass_filter()`.'
                            'If the recording is already filtered, you can also do `recording.annotate(is_filtered=True)')

        self.recording = recording
        self.sorting = sorting
        self.folder = Path(folder)

        # cache in memory
        self._waveforms = {}
        self._template_average = {}
        self._template_median = {}
        self._params = {}

        if (self.folder / 'params.json').is_file():
            with open(str(self.folder / 'params.json'), 'r') as f:
                self._params = json.load(f)

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.recording.get_num_segments()
        nchan = self.recording.get_num_channels()
        nunits = self.sorting.get_num_units()
        txt = f'{clsname}: {nchan} channels - {nunits} units - {nseg} segments'
        if len(self._params) > 0:
            max_spikes_per_unit = self._params['max_spikes_per_unit']
            txt = txt + f'\n  before:{self.nbefore} after{self.nafter} n_per_units: {max_spikes_per_unit}'
        return txt

    @classmethod
    def load_from_folder(cls, folder):
        folder = Path(folder)
        recording = load_extractor(folder / 'recording.json')
        sorting = load_extractor(folder / 'sorting.json')
        we = cls(recording, sorting, folder)
        return we

    @classmethod
    def create(cls, recording, sorting, folder, remove_if_exists=False):
        folder = Path(folder)
        if folder.is_dir():
            if remove_if_exists:
                shutil.rmtree(folder)
            else:
                raise FileExistsError('Folder already exists')
        folder.mkdir(parents=True)

        if recording.is_dumpable:
            recording.dump(folder / 'recording.json', relative_to=None)
        if sorting.is_dumpable:
            sorting.dump(folder / 'sorting.json', relative_to=None)

        return cls(recording, sorting, folder)

    def _reset(self):
        self._waveforms = {}
        self._template_average = {}
        self._template_median = {}
        self._params = {}

        waveform_folder = self.folder / 'waveforms'
        if waveform_folder.is_dir():
            shutil.rmtree(waveform_folder)
        waveform_folder.mkdir()

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
            if not self.recording.has_scaled_traces():
                print("Setting 'return_scaled' to False")
                return_scaled = False

        if np.issubdtype(dtype, np.integer) and return_scaled:
            dtype = "float32"

        if max_spikes_per_unit is not None:
            max_spikes_per_unit = int(max_spikes_per_unit)

        self._params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=max_spikes_per_unit,
            return_scaled=return_scaled,
            dtype=dtype.str)

        (self.folder / 'params.json').write_text(
            json.dumps(check_json(self._params), indent=4), encoding='utf8')

    @property
    def nbefore(self):
        sampling_frequency = self.recording.get_sampling_frequency()
        nbefore = int(self._params['ms_before'] * sampling_frequency / 1000.)
        return nbefore

    @property
    def nafter(self):
        sampling_frequency = self.recording.get_sampling_frequency()
        nafter = int(self._params['ms_after'] * sampling_frequency / 1000.)
        return nafter

    @property
    def nsamples(self):
        return self.nbefore + self.nafter

    @property
    def return_scaled(self):
        return self._params['return_scaled']

    def get_waveforms(self, unit_id, with_index=False):
        """
        Return waveforms

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_channels, num_samples)
        indices: np.array
            If 'with_index' is True, the spike indices corresponding to the waveforms extracted
        """
        assert unit_id in self.sorting.unit_ids, "'unit_id' is invalid"

        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            waveform_file = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
            if not waveform_file.is_file():
                raise Exception('waveforms not extracted yet : please do WaveformExtractor.run() fisrt')

            wfs = np.load(waveform_file)
            self._waveforms[unit_id] = wfs

        if with_index:
            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            sampled_index = np.load(sampled_index_file)
            return wfs, sampled_index
        else:
            return wfs

    def get_template(self, unit_id, mode='median'):
        """
        Return template (average waveform)

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        mode: str
            'mean' or 'median' (default)

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """
        assert mode in ('median', 'average')
        assert unit_id in self.sorting.unit_ids

        if mode == 'median':
            if unit_id in self._template_median:
                return self._template_median[unit_id]
            else:
                wfs = self.get_waveforms(unit_id)
                template = np.median(wfs, axis=0)
                self._template_median[unit_id] = template
                return template
        elif mode == 'average':
            if unit_id in self._template_average:
                return self._template_average[unit_id]
            else:
                wfs = self.get_waveforms(unit_id)
                template = np.average(wfs, axis=0)
                self._template_average[unit_id] = template
                return template

    def get_all_templates(self, unit_ids=None, mode='median'):
        """
        Return several templates (average waveform)

        Parameters
        ----------
        unit_ids: list or None
            Unit ids to retrieve waveforms for
        mode: str
            'mean' or 'median' (default)

        Returns
        -------
        templates: np.array
            The returned templates (num_units, num_samples, num_channels)
        """
        if unit_ids is None:
            unit_ids = self.sorting.unit_ids
        num_chans = self.recording.get_num_channels()

        dtype = self._params['dtype']
        templates = np.zeros((len(unit_ids), self.nsamples, num_chans), dtype=dtype)
        for i, unit_id in enumerate(unit_ids):
            templates[i, :, :] = self.get_template(unit_id, mode=mode)
        return templates

    def sample_spikes(self):
        p = self._params
        nbefore = self.nbefore
        nafter = self.nafter

        selected_spikes = select_random_spikes_uniformly(self.recording, self.sorting,
                                                         self._params['max_spikes_per_unit'], nbefore, nafter)

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

            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            np.save(sampled_index_file, sampled_index)

        return selected_spikes

    def run(self, **job_kwargs):
        p = self._params
        sampling_frequency = self.recording.get_sampling_frequency()
        num_chans = self.recording.get_num_channels()
        nbefore = self.nbefore
        nafter = self.nafter
        return_scaled = self.return_scaled

        n_jobs = ensure_n_jobs(self.recording, job_kwargs.get('n_jobs', None))

        selected_spikes = self.sample_spikes()

        # get spike times
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                selected_spike_times[unit_id].append(spike_times[sel])

        # prepare memmap
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
            n_spikes = np.sum([e.size for e in selected_spike_times[unit_id]])
            shape = (n_spikes, self.nsamples, num_chans)
            wfs = np.zeros(shape, dtype=p['dtype'])
            np.save(file_path, wfs)
            # wfs = np.load(file_path, mmap_mode='r+')
            wfs_memmap[unit_id] = file_path

        # and run
        func = _waveform_extractor_chunk
        init_func = _init_worker_waveform_extractor
        if n_jobs == 1:
            init_args = (self.recording, self.sorting,)
        else:
            init_args = (self.recording.to_dict(), self.sorting.to_dict(),)
        init_args = init_args + (wfs_memmap, selected_spikes, selected_spike_times, nbefore, nafter, return_scaled)
        processor = ChunkRecordingExecutor(self.recording, func, init_func, init_args, job_name='extract waveforms',
                                           **job_kwargs)
        processor.run()


def select_random_spikes_uniformly(recording, sorting, max_spikes_per_unit, nbefore=None, nafter=None):
    """
    Uniform random selection of spike across segment per units.
    
    This function does not select spikes near border if nbefore/nafter are not None.
    """
    unit_ids = sorting.unit_ids
    num_seg = sorting.get_num_segments()

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

            if nbefore is not None:
                # clean border
                assert nafter is not None
                spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sampled_spike_times = spike_times[inds]
                num_samples = recording.get_num_samples(segment_index=segment_index)
                mask = (sampled_spike_times >= nbefore) & (sampled_spike_times < (num_samples - nafter))
                inds = inds[mask]

            sel_spikes.append(inds)
        selected_spikes[unit_id] = sel_spikes
    return selected_spikes


# used by WaveformExtractor + ChunkRecordingExecutor
def _init_worker_waveform_extractor(recording, sorting, wfs_memmap,
                                    selected_spikes, selected_spike_times, nbefore, nafter, return_scaled):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording

    if isinstance(sorting, dict):
        from spikeinterface.core import load_extractor
        sorting = load_extractor(sorting)
    worker_ctx['sorting'] = sorting

    worker_ctx['wfs_memmap_files'] = wfs_memmap
    worker_ctx['selected_spikes'] = selected_spikes
    worker_ctx['selected_spike_times'] = selected_spike_times
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['return_scaled'] = return_scaled

    num_seg = sorting.get_num_segments()
    unit_cum_sum = {}
    for unit_id in sorting.unit_ids:
        # spike per segment
        n_per_segment = [selected_spikes[unit_id][i].size for i in range(num_seg)]
        cum_sum = [0] + np.cumsum(n_per_segment).tolist()
        unit_cum_sum[unit_id] = cum_sum
    worker_ctx['unit_cum_sum'] = unit_cum_sum

    return worker_ctx


# used by WaveformExtractor + ChunkRecordingExecutor
def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    sorting = worker_ctx['sorting']
    wfs_memmap_files = worker_ctx['wfs_memmap_files']
    selected_spikes = worker_ctx['selected_spikes']
    selected_spike_times = worker_ctx['selected_spike_times']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    return_scaled = worker_ctx['return_scaled']
    unit_cum_sum = worker_ctx['unit_cum_sum']

    to_extract = {}
    for unit_id in sorting.unit_ids:
        spike_times = selected_spike_times[unit_id][segment_index]
        i0 = np.searchsorted(spike_times, start_frame)
        i1 = np.searchsorted(spike_times, end_frame)
        if i0 != i1:
            to_extract[unit_id] = i0, i1, spike_times[i0:i1]

    if len(to_extract) > 0:
        start = min(st[0] for _, _, st in to_extract.values()) - nbefore
        end = max(st[-1] for _, _, st in to_extract.values()) + nafter
        start = int(start)
        end = int(end)

        # load trace in memory
        traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index,
                                      return_scaled=return_scaled)

        for unit_id, (i0, i1, local_spike_times) in to_extract.items():
            wfs = np.load(wfs_memmap_files[unit_id], mmap_mode="r+")
            for i in range(local_spike_times.size):
                st = local_spike_times[i]
                st = int(st)
                pos = unit_cum_sum[unit_id][segment_index] + i0 + i
                wfs[pos, :, :] = traces[st - start - nbefore:st - start + nafter, :]


def extract_waveforms(recording, sorting, folder,
                      load_if_exists=False,
                      ms_before=3., ms_after=4.,
                      max_spikes_per_unit=500,
                      overwrite=False,
                      return_scaled=True,
                      dtype=None,
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
    folder: str or Path
        The folder where waveforms are cached
    load_if_exists: bool
        If True and waveforms have already been extracted in the specified folder, they are loaded
        and not recomputed.
    ms_before: float
        Time in ms to cut before spike peak
    ms_after: float
        Time in ms to cut after spike peak
    max_spikes_per_unit: int or None
        Number of spikes per unit to extract waveforms from (default 500).
        Use None to extract waveforms for all spikes
    overwrite: bool
        If True and 'folder' exists, the folder is removed and waveforms are recomputed.
        Othewise an error is raised.
    return_scaled: bool
        If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV.
    dtype: dtype or None
        Dtype of the output waveforms. If None, the recording dtype is maintained.

    {}

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object

    """
    folder = Path(folder)
    assert not (overwrite and load_if_exists), "Use either 'overwrite=True' or 'load_if_exists=True'"
    if overwrite and folder.is_dir():
        shutil.rmtree(folder)
    if load_if_exists and folder.is_dir():
        we = WaveformExtractor.load_from_folder(folder)
    else:
        we = WaveformExtractor.create(recording, sorting, folder)
        we.set_params(ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=max_spikes_per_unit, dtype=dtype,
                      return_scaled=return_scaled)
        we.run(**job_kwargs)

    return we


extract_waveforms.__doc__ = extract_waveforms.__doc__.format(_shared_job_kwargs_doc)
