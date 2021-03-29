from pathlib import Path
import shutil
import json

import numpy as np

from .base import load_extractor

from .core_tools import check_json
from .job_tools import ChunkRecordingExecutor


class WaveformExtractor:
    """
    Class to extract waveform on a recording from a sorting.
    Waveforms are persistent on disk + cached in memory.
    This allow fast acces.
    
    Usage :
    
    # create
    we = WaveformExtractor.create(recording, sorting, folder)
    
    # compute
    we = we.set_params(...)
    we = we.run(...)
    
    # retrieve
    waveforms = we.get_waveforms(unit_id)
    template = we.get_template(unit_id, mode='median')
    
    # Load persistent from folder (in another session
    WaveformExtractor.load_from_folder(folder)
    
    """
    def __init__(self, recording, sorting, folder):

        assert recording.get_num_segments() == sorting.get_num_segments(), \
            'WaveformExtractor : is it a joke ?'

        np.testing.assert_almost_equal(recording.get_sampling_frequency(),
            sorting.get_sampling_frequency(), decimal=2)


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
                self._params =  json.load(f)

    @classmethod
    def load_from_folder(cls, folder):
        folder = Path(folder)
        recording = load_extractor(folder /  'recording.json')
        sorting = load_extractor(folder /  'sorting.json')
        we = cls(recording, sorting, folder)
        return we

    @classmethod
    def create(cls,  recording, sorting, folder):
        folder = Path(folder)
        if folder.is_dir():
            raise valueError('Folder already exssts')
        folder.mkdir()
        
        recording.dump(folder / 'recording.json')
        sorting.dump(folder / 'sorting.json')
        
        return cls( recording, sorting, folder)

    def _reset(self):
        self._waveforms = {}
        self._template_average = {}
        self._template_median = {}
        self._params = {}
        
        waveform_folder = self.folder / 'waveforms'
        if waveform_folder.is_dir():
            shutil.rmtree(waveform_folder)
        waveform_folder.mkdir()
        

    def set_params(self, ms_before=3., ms_after=4., max_spikes_per_unit=500, dtype=None):
        """
        
        """
        self._reset()
        
        if dtype is None:
            dtype = self.recording.get_dtype()
        
        self._params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=int(max_spikes_per_unit),
            dtype=dtype.str)
        
        (self.folder / 'params.json').write_text(
                json.dumps(check_json(self._params), indent=4), encoding='utf8')
    
    def get_waveforms(self, unit_id, with_index=False):
        """
        
        """
        assert unit_id in self.sorting.unit_ids
        
        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            waveform_file = self.folder / 'waveforms' / f'waveforms_{unit_id}.raw'
            if not waveform_file.is_file():
                raise Exception('waveforms not extracted yet : please do WaveformExtractor.run() fisrt')
            
            p = self._params
            sampling_frequency = self.recording.get_sampling_frequency()
            num_chans = self.recording.get_num_channels()
            
            before = int(p['ms_before'] * sampling_frequency / 1000.)
            after = int(p['ms_after'] * sampling_frequency / 1000.)
            
            wfs = np.memmap(str(waveform_file), dtype=p['dtype']).reshape(-1, before + after, num_chans)
            # get a copy to have a memory faster access and avoid write back in file
            wfs = wfs.copy()
            self._waveforms[unit_id] = wfs
        
        if  with_index:
            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            sampled_index = np.load(sampled_index_file)
            return wfs, sampled_index
        else:
            return wfs

    def get_template(self, unit_id, mode='median'):
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

    
    def sample_spikes(self):
        p = self._params
        sampling_frequency = self.recording.get_sampling_frequency()
        before = int(p['ms_before'] * sampling_frequency  / 1000.)
        after = int(p['ms_after'] * sampling_frequency  / 1000.)
        width = before + after
        
        selected_spikes = select_random_spikes(self.recording, self.sorting, self._params['max_spikes_per_unit'], before, after)
        
        # store in a 2 columns (spike_index, segment_index) in a npy file
        for unit_id in self.sorting.unit_ids:
            
            
            n = np.sum([e.size for e in selected_spikes[unit_id]])
            sampled_index = np.zeros(n, dtype=[('spike_index', 'int64'), ('segment_index', 'int64')])
            pos = 0
            for segment_index in range(self.sorting.get_num_segments()):
                inds = selected_spikes[unit_id][segment_index]
                sampled_index[pos:pos+inds.size]['spike_index'] = inds
                sampled_index[pos:pos+inds.size]['segment_index'] = segment_index
                pos += inds.size
                
            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            np.save(sampled_index_file, sampled_index)
        
        return selected_spikes
        
    
    def run(self, **job_kwargs):
        p = self._params
        sampling_frequency = self.recording.get_sampling_frequency()
        num_chans = self.recording.get_num_channels()
        before = int(p['ms_before'] * sampling_frequency  / 1000.)
        after = int(p['ms_after'] * sampling_frequency  / 1000.)
        width = before + after
    
        selected_spikes = self.sample_spikes()
        #~ selected_spikes = select_random_spikes(self.recording, self.sorting, self._params['max_spikes_per_unit'], before, after)
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                selected_spike_times[unit_id].append(spike_times[sel])
        
        
        
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.folder / 'waveforms' / f'waveforms_{unit_id}.raw'
            n_spikes = np.sum([ e.size for e in selected_spike_times[unit_id] ])
            shape = (n_spikes, width, num_chans)
            wfs = np.memmap(str(file_path), dtype=p['dtype'], mode='w+', shape=shape)
            wfs_memmap[unit_id] = wfs
        

        func = _waveform_extractor_chunk
        init_func = _init_worker_waveform_extractor
        init_args = (self.recording.to_dict(), self.sorting.to_dict(), wfs_memmap,
                selected_spikes, selected_spike_times, before, after)
        
        processor = ChunkRecordingExecutor(self.recording, func, init_func, init_args, **job_kwargs)
        processor.run()


# TODO maybe move this in a better place (in core!!)
def select_random_spikes(recording, sorting, max_spikes_per_unit, before=None, after=None):
    """
    Uniform random selection of spike across segment per units.
    
    Do not select spike near border is before/after not None
    
    """
    unit_ids = sorting.unit_ids
    num_seg = sorting.get_num_segments()
    
    selected_spikes = {}
    for unit_id in unit_ids:
        # spike per segment
        n_per_segment = [sorting.get_unit_spike_train(unit_id, segment_index=i).size for i in range(num_seg)]
        cum_sum = [0] + np.cumsum(n_per_segment).tolist()
        total = np.sum(n_per_segment)
        if total > max_spikes_per_unit:
            global_inds = np.random.choice(total, size=max_spikes_per_unit, replace=False)
            global_inds = np.sort(global_inds)
        else:
            global_inds = np.arange(total)
        sel_spikes = []
        for segment_index in range(num_seg):
            in_segment = (global_inds>=cum_sum[segment_index]) & (global_inds < cum_sum[segment_index+1])
            inds = global_inds[in_segment] - cum_sum[segment_index]
            
            if before is not None:
                # clean border
                assert after is not None
                spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sampled_spike_times = spike_times[inds]
                num_samples = recording.get_num_samples(segment_index=segment_index)
                mask = (sampled_spike_times >=before) & (sampled_spike_times<(num_samples-after))
                inds = inds[mask]
            
            sel_spikes.append(inds)
        selected_spikes[unit_id] = sel_spikes
    
    return selected_spikes


# used by WaveformExtractor + ChunkRecordingExecutor
def _init_worker_waveform_extractor(recording, sorting, wfs_memmap,
            selected_spikes, selected_spike_times, before, after):
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

    worker_ctx['wfs_memmap'] = wfs_memmap
    worker_ctx['selected_spikes'] = selected_spikes
    worker_ctx['selected_spike_times'] = selected_spike_times
    worker_ctx['before'] = before
    worker_ctx['after'] = after
    
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
    wfs_memmap = worker_ctx['wfs_memmap']
    selected_spikes = worker_ctx['selected_spikes']
    selected_spike_times = worker_ctx['selected_spike_times']
    before = worker_ctx['before']
    after = worker_ctx['after']
    unit_cum_sum = worker_ctx['unit_cum_sum']
    # print('before', before, 'after', after, type(before), type(after))
    
    # print('_waveform_extractor_chunk', segment_index, start_frame, end_frame)
    to_extract = {}
    for unit_id in sorting.unit_ids:
        spike_times = selected_spike_times[unit_id][segment_index]
        i0 = np.searchsorted(spike_times, start_frame)
        i1 = np.searchsorted(spike_times, end_frame)
        #~ print(unit_id, i0, i1)
        if i0 != i1:
            to_extract[unit_id] = i0, i1, spike_times[i0:i1]
        #~ print(to_extract)
    
    if len(to_extract) > 0:
        start = min(st[0] for _, _, st in to_extract.values()) - before
        end = max(st[-1] for _, _, st in to_extract.values()) + after
        start = int(start)
        end = int(end)
        
        # load trace in memory
        traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index)
        
        for unit_id, (i0, i1, spike_times) in to_extract.items():
            wfs = wfs_memmap[unit_id]
            for i in range(spike_times.size):
                st = spike_times[i]
                st = int(st)
                pos = unit_cum_sum[unit_id][segment_index] + i0 + i
                # print('unit_id', unit_id, 'pos', pos)
                # print(st, st - start, st - start - before)
                # print(traces[st - start - before:st - start + after, :].shape)
                wfs[pos, :, :] = traces[st - start - before:st - start + after, :]

