from typing import List, Union
from spikeinterface.core.mytypes import ChannelId, SampleIndex, ChannelIndex, Order, SamplingFrequencyHz

from pathlib import Path
from typing import Union

import numpy as np

import neo
from .neobaseextractor import NeoBaseRecordingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading axona raw data (.bin file and .set file)

    Based on neo.rawio.AxonaRawIO

    Parameters
    ----------
    file_path: str or Path
        Full filename of the .set or .bin file.
    """
    extractor_name = 'AxonaRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': str(file_path)}
        super().__init__(**neo_kwargs)
        self._main_ids = np.array([int(el) for el in self._main_ids])

    # Overwrite bc prefer_slice=True breaks AxonaRawIO._get_analogsignal_chunk()
    def get_traces(self,
                   segment_index: Union[int, None] = None,
                   start_frame: Union[SampleIndex, None] = None,
                   end_frame: Union[SampleIndex, None] = None,
                   channel_ids: Union[List[ChannelId], None] = None,
                   order: Union[Order, None] = None,
                   return_scaled=False,
                   ):
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=False)
        rs = self._recording_segments[segment_index]
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            traces = np.asanyarray(traces, order=order)
        if return_scaled:
            gains = self.get_property('gain_to_uV')
            offsets = self.get_property('offset_to_uV')
            if gains is None or offsets is None:
                raise ValueError('This recording do not support return_scaled=True (need gain_to_uV and offset_to_uV properties)')
            gains = gains[channel_indices].astype('float32')
            offsets = offsets[channel_indices].astype('float32')
            traces = traces.astype('float32') * gains + offsets
        return traces


class AxonaUnitRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Instantiates a RecordinExtractor from an Axon Unit mode file.
    Since the unit mode format only saves waveform cutouts, the get_traces
    function fills in the rest of the recording with Gaussian uncorrelated noise

    Parameters
    ----------
    file_path: str or Path
        Full filename of the .set or .bin file.
    noise_std: float
        Standard deviation of the Gaussian background noise
    """
    extractor_name = 'AxonaUnitRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, file_path, noise_std: float = 3, **kargs):
        neo_kwargs = {'filename': str(file_path)}
        super().__init__(**neo_kwargs)
        self._noise_std = noise_std

    def get_traces(self, segment_index=0, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):

        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples(segment_index)

        timebase_sr = int(self.neo_reader.file_parameters['unit']['timebase'].split(' ')[0])
        samples_pre = int(self.neo_reader.file_parameters['set']['file_header']['pretrigSamps'])
        samples_post = int(self.neo_reader.file_parameters['set']['file_header']['spikeLockout'])
        sampling_rate = self.get_sampling_frequency()

        tcmap = self._get_tetrode_channel_table(channel_ids)

        traces = self._noise_std * np.random.randn(len(channel_ids), end_frame - start_frame)

        # Loop through tetrodes and include requested channels in traces
        itrc = 0
        for tetrode_id in np.unique(tcmap[:, 0]):

            channels_oi = tcmap[tcmap[:, 0] == tetrode_id, 2]

            waveforms = self.neo_reader._get_spike_raw_waveforms(
                block_index=0, seg_index=0,
                unit_index=tetrode_id - 1,  # Tetrodes IDs are 1-indexed
                t_start=start_frame / sampling_rate,
                t_stop=end_frame / sampling_rate
            )
            waveforms = waveforms[:, channels_oi, :]
            nch = len(channels_oi)

            spike_train = self.neo_reader._get_spike_timestamps(
                block_index=0, seg_index=0,
                unit_index=tetrode_id - 1,
                t_start=start_frame / sampling_rate,
                t_stop=end_frame / sampling_rate
            )

            # Fill waveforms into traces timestamp by timestamp
            for t, wf in zip(spike_train, waveforms):

                t = int(t // (timebase_sr / sampling_rate))  # timestamps are sampled at higher frequency
                t = t - start_frame
                if t - samples_pre < 0:
                    traces[itrc:itrc + nch, :t + samples_post] = wf[:, samples_pre - t:]
                elif t + samples_post > traces.shape[1]:
                    traces[itrc:itrc + nch, t - samples_pre:] = wf[:, :traces.shape[1] - (t - samples_pre)]
                else:
                    traces[itrc:itrc + nch, t - samples_pre:t + samples_post] = wf

            itrc += nch

        return traces.T

    def get_num_frames(self):
        n = self.neo_reader.get_signal_size(self.block_index, self.seg_index, stream_index=0)
        if self.get_sampling_frequency() == 24000:
            n = n // 2
        return n

    def get_sampling_frequency(self):
        return int(self.neo_reader.file_parameters['unit']['sample_rate'].split(' ')[0])

    def _get_tetrode_channel_table(self, channel_ids):
        '''Create auxiliary np.array with the following columns:
        Tetrode ID, Channel ID, Channel ID within tetrode
        This is useful in `get_traces()`
        Parameters
        ----------
        channel_ids : list
            List of channel ids to include in table
        Returns
        -------
        np.array
            Rows = channels,
            columns = TetrodeID, ChannelID, ChannelID within Tetrode
        '''
        channel_ids = [int(el) for el in channel_ids]
        active_tetrodes = self.neo_reader.get_active_tetrode()

        tcmap = np.zeros((len(active_tetrodes) * 4, 3), dtype=int)
        row_id = 0
        for tetrode_id in [int(s[0].split(' ')[1]) for s in self.neo_reader.header['spike_channels']]:

            all_channel_ids = self.neo_reader._get_channel_from_tetrode(tetrode_id)

            for i in range(4):
                tcmap[row_id, 0] = int(tetrode_id)
                tcmap[row_id, 1] = int(all_channel_ids[i])
                tcmap[row_id, 2] = int(i)
                row_id += 1

        del_idx = [False if i in channel_ids else True for i in tcmap[:, 1]]

        return np.delete(tcmap, del_idx, axis=0)
