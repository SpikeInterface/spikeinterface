import numpy as np
from spikeinterface import ChannelsAggregationRecording 
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

class MaskOutArtifactsRecording(BasePreprocessor):
    """
    Removes  high power chunks from recording extractor traces. This is only 
    recommended for traces that are centered around zero (e.g. through a 
    prior highpass filter).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    threshold: float
        Threshold in standar desviations (MAD) over the mean power to detect extereme chunks.
    chunk_size: int
        Size of the chunks to segment the signal and calculate power.
    by_property: None or str
        Mask out chunks in the splited recording if any of its channels is over the threshold.
    Returns
    -------
    masked_out_artifacts: MaskOutArtifactsRecording
        The MaskOutArtifactsRecording extractor object
    '''
    """
    name = 'mask_out_artifacts'

    def __init__(self, recording, threshold=5, chunk_size=2000, thr_prop=0.15):

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            rec_segment = MaskOutArtifactsRecordingSegment(parent_segment, threshold=threshold,
                            chunk_size=chunk_size, thr_prop=thr_prop)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), threshold=threshold,
                            chunk_size=chunk_size, thr_prop=thr_prop)


class MaskOutArtifactsRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, threshold, chunk_size, thr_prop):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.chunk_size = chunk_size

        self.M = int(parent_recording_segment.parent_extractor.get_num_channels())
        self.N = int(self.get_num_samples())
        self.num_chunks = np.ceil(self.N / chunk_size).astype(int)
        nsubs = int(np.ceil(self.num_chunks*thr_prop))
        norms = np.zeros((self.M, nsubs))  # num channels x num_chunks
        random_chunk = np.random.choice(np.arange(self.num_chunks), size=nsubs, replace=False)
        for i, chunk_i in enumerate(random_chunk):
            t1 = chunk_i * chunk_size  # first timepoint of the chunk
            t2 = np.minimum(self.N, (t1 + chunk_size))  # last timepoint of chunk (+1)            
            chunk = parent_recording_segment.get_traces(start_frame=t1, end_frame=t2,channel_indices=None).astype('float32') # Read the chunk
            norms[:, i] = np.sqrt(np.sum(chunk ** 2, axis=0))  # num_channels x num_chunks

        # determine which chunks to use
        self.maskout = np.zeros(self.num_chunks).astype(np.bool) 
        self.calculated = np.zeros(self.num_chunks).astype(np.bool)

        self.thrs = np.zeros(self.M)  # num channels x num_chunks

        for m in np.arange(self.M):
            vals = norms[m, :]
            mu = np.mean(vals, axis=0)
            sigma =np.median(np.abs(vals - mu), axis=0) * 1.4826
            self.thrs[m] = mu + sigma * threshold
            chunks2mask = random_chunk[norms[m, :] > self.thrs[m]]
            self.maskout[chunks2mask] = True
        self.calculated[random_chunk] = True

    def get_traces(self, start_frame, end_frame, channel_indices):

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        #che sttus of fist and last maskchunks
        start_mc = int(np.floor(start_frame/self.chunk_size))
        end_mc =  int(np.floor(end_frame/self.chunk_size))
        if self.calculated[start_mc]:
             extra_start_frame = 0
        else:
            extra_start_frame = start_frame  - (start_mc * self.chunk_size)

        if self.calculated[end_mc]:
            extra_end_frame = 0
        else:
            extra_end_frame = np.minimum(self.N,(end_mc+1) * self.chunk_size-1) - end_frame

        start_frame -= extra_start_frame
        end_frame += extra_end_frame
        l = end_frame - start_frame
        load_all_channels = ~np.all(self.calculated[int(np.floor((start_frame) / self.chunk_size)):int(np.floor((end_frame) / self.chunk_size))+1])
        
        if load_all_channels:
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices=None)
        else:
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices=channel_indices)
        traces = traces.copy()
        s = 0
        while(s<=l):
            mc = int(np.floor((start_frame + s) / self.chunk_size))
            if not self.calculated[mc]:
                norms = np.sqrt(np.sum(traces[s:s+self.chunk_size,:].astype('float32') ** 2, axis=0)) 
                for m in np.arange(self.M):
                    if  norms[m] > self.thrs[m]:
                        self.maskout[mc] = True
                        break
                self.calculated[mc]=True

            if self.maskout[mc]:
                traces[s:s+self.chunk_size,:] = 0
            s += self.chunk_size

        return traces[extra_start_frame:-(1+extra_end_frame),channel_indices]


# function for API
def mask_out_artifacts(recording, by_property=None, *args, **kwargs):
    '''
    Find and remove chunks of the signal with extereme power.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    threshold: float
        Threshold in standar desviations (MAD) over the mean power to detect extereme chunks.
    chunk_size: int
        Size of the chunks to segment the signal and calculate power.
    by_property: None or str
        Mask out chunks in the splited recording if any of its channels is over the threshold.
    Returns
    -------
    masked_out_artifacts: MaskOutArtifactsRecording
        The MaskOutArtifactsRecording extractor object
    '''

    if by_property is None:
        rec = MaskOutArtifactsRecording(recording, *args, **kwargs)
    else:
        rec_list = [MaskOutArtifactsRecording(r, *args, **kwargs) for r in recording.split_by(property=by_property, outputs='list')]
        rec_list_ids = np.concatenate([r.get_channel_ids() for r in rec_list])
        rec = ChannelsAggregationRecording(rec_list, rec_list_ids)
    return rec