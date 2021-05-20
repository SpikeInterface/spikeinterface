from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import write_binary_recording

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .utils.mdaio import DiskReadMda, readmda, writemda64, MdaHeader


class MdaRecording(BaseRecording):
    extractor_name = 'MdaRecording'
    has_default_locations = True
    has_unscaled = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path, raw_fname='raw.mda', params_fname='params.json', geom_fname='geom.csv'):
        folder_path = Path(folder_path)
        self._folder_path = folder_path
        self._dataset_params = read_dataset_params(self._folder_path, params_fname)
        self._timeseries_path = self._folder_path / raw_fname
        geom = np.loadtxt(self._folder_path / geom_fname, delimiter=',', ndmin=2)
        self._diskreadmda = DiskReadMda(str(self._timeseries_path))
        dtype = self._diskreadmda.dt()
        num_channels = self._diskreadmda.N1()
        assert geom.shape[0] == self._diskreadmda.N1(), f'Incompatible dimensions between geom.csv and timeseries ' \
                                                        f'file: {geom.shape[0]} <> {self._diskreadmda.N1()}'
        BaseRecording.__init__(self, sampling_frequency=self._dataset_params['samplerate'] * 1.0,
                               channel_ids=np.arange(num_channels), dtype=dtype)
        rec_segment = MdaRecordingSegment(self._diskreadmda)
        self.add_recording_segment(rec_segment)
        self.set_channel_locations(geom)
        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'raw_fname': raw_fname, 'params_fname': params_fname,
                        'geom_fname': geom_fname}

    @staticmethod
    def write_recording(recording, save_path, params=dict(), raw_fname='raw.mda', params_fname='params.json',
                        geom_fname='geom.csv', verbose=True, dtype=None, **job_kwargs):
        """
        Writes recording to file in MDA format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be saved
        save_path: str or Path
            The folder in which the Mda files are saved
        params: dictionary
            Dictionary with optional parameters to save metadata. Sampling frequency is appended to this dictionary.
        raw_fname: str
            File name of raw file (default raw.mda)
        params_fname: str
            File name of params file (default params.json)
        geom_fname: str
            File name of geom file (default geom.csv)
        dtype: dtype
            dtype to be used. If None dtype is same as recording traces.
        verbose: bool
            If True, output is verbose
        **job_kwargs:
            Use by job_tools modules to set:
                * chunk_size or chunk_memory, or total_memory
                * n_jobs
                * progress_bar
        """
        assert recording.get_num_segments() == 1, "MdaRecording.write_recording() can only write a single segment " \
                                                  "recording"
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file_path = save_path / raw_fname
        parent_dir = save_path
        num_chan = recording.get_num_channels()
        num_frames = recording.get_num_frames()

        geom = recording.get_channel_locations()

        if dtype is None:
            dtype = recording.get_dtype()

        if dtype == 'float':
            dtype = 'float32'
        if dtype == 'int':
            dtype = 'int16'

        with save_file_path.open('wb') as f:
            header = MdaHeader(dt0=dtype, dims0=(num_chan, num_frames))
            header.write(f)
            header_size = header.header_size

        write_binary_recording(recording, files_path=save_path, dtype=dtype,
                               byte_offset=header_size, verbose=verbose, **job_kwargs)

        params["samplerate"] = float(recording.get_sampling_frequency())
        with (parent_dir / params_fname).open('w') as f:
            json.dump(params, f)
        np.savetxt(str(parent_dir / geom_fname), geom, delimiter=',')


class MdaRecordingSegment(BaseRecordingSegment):
    def __init__(self, diskreadmda):
        self._diskreadmda = diskreadmda
        BaseRecordingSegment.__init__(self)
        self._num_samples = self._diskreadmda.N2()

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(self,
                   start_frame=None,
                   end_frame=None,
                   channel_indices=None,
                   ) -> np.ndarray:
        recordings = self._diskreadmda.readChunk(i1=0, i2=start_frame, N1=X.N1(), N2=end_frame - start_frame)
        recordings = recordings[channel_indices, :].T
        return recordings


# For backward compatibity (old good time)
MdaRecordingExtractor = MdaRecording


class MdaSorting(BaseSorting):
    extractor_name = 'MdaSorting'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path, sampling_frequency):
        firings = readmda(str(file_path))
        labels = firings[2, :]
        unit_ids = np.unique(labels).astype(int)
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)

        sorting_segment = MdaSortingSegment(firings)
        self.add_sorting_segment(sorting_segment)

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency}


class MdaSortingSegment(BaseSortingSegment):
    def __init__(self, firings):
        self._firings = firings
        self._max_channels = self._firings[0, :]
        self._spike_times = self._firings[1, :]
        self._labels = self._firings[2, :]
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame,
                             end_frame,
                             ) -> np.ndarray:
        # must be implemented in subclass
        inds = np.where(
            (self._labels == unit_id) & (start_frame <= self._spike_times) & (self._spike_times < end_frame))
        return np.rint(self._spike_times[inds]).astype(int)


    @staticmethod
    def write_sorting(sorting, save_path, write_primary_channels=False):
        assert sorting.get_num_segments() == 1, "MdaSorting.write_sorting() can only write a single segment " \
                                                "sorting"
        unit_ids = sorting.get_unit_ids()
        times_list = []
        labels_list = []
        primary_channels_list = []
        for unit_id in unit_ids:
            times = sorting.get_unit_spike_train(unit_id=unit_id)
            times_list.append(times)
            labels_list.append(np.ones(times.shape) * unit_id)
            if write_primary_channels:
                if 'max_channel' in sorting.get_unit_property_names(unit_id):
                    primary_channels_list.append([sorting.get_unit_property(unit_id, 'max_channel')] * times.shape[0])
                else:
                    raise ValueError(
                        "Unable to write primary channels because 'max_channel' spike feature not set in unit " + str(
                            unit_id))
            else:
                primary_channels_list.append(np.zeros(times.shape))
        all_times = _concatenate(times_list)
        all_labels = _concatenate(labels_list)
        all_primary_channels = _concatenate(primary_channels_list)
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        all_primary_channels = all_primary_channels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[0, :] = all_primary_channels
        firings[1, :] = all_times
        firings[2, :] = all_labels

        writemda64(firings, save_path)

# For backward compatibity (old good time)
MdaSortingExtractor = MdaSorting


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def read_dataset_params(dsdir, params_fname):
    fname1 = dsdir / params_fname
    if not fname1.is_file():
        raise Exception('Dataset parameter file does not exist: ' + fname1)
    with open(fname1) as f:
        return json.load(f)
