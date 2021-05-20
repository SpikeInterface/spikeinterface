from spikeinterface.core import BinaryRecordingExtractor, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import write_binary_recording
from probeinterface import read_prb

import json
import numpy as np
from pathlib import Path

try:
    import hybridizer.io as sbio
    import hybridizer.probes as sbprb
    import yaml

    HAVE_SBEX = True
except ImportError:
    HAVE_SBEX = False


class SHYBRIDRecordingExtractor(BinaryRecordingExtractor):
    extractor_name = 'MdaRecording'
    has_default_locations = True
    has_unscaled = False
    installed = HAVE_SBEX  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the SHYBRID extractors, install SHYBRID and pyyaml: " \
                        "\n\n pip install shybrid pyyaml\n\n"

    def __init__(self, file_path):
        # load params file related to the given shybrid recording
        assert self.installed, self.installation_mesg
        params = sbio.get_params(file_path)['data']

        # create a shybrid probe object
        probe = sbprb.Probe(params['probe'])
        nb_channels = probe.total_nb_channels

        # translate the byte ordering
        # TODO still ambiguous, shybrid should assume time_axis=1, since spike interface makes an assumption on the byte ordering
        byte_order = params['order']
        if byte_order == 'C':
            time_axis = 1
        elif byte_order == 'F':
            time_axis = 0

        # piggyback on binary data recording extractor
        BinaryRecordingExtractor.__init__(self,
                                          files_path=file_path,
                                          sampling_frequency=params['fs'],
                                          num_chan=nb_channels,
                                          dtype=params['dtype'],
                                          time_axis=time_axis)

        # load probe file
        probe = read_prb(params['probe'])
        self.set_probe(probe, in_place=True)
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    @staticmethod
    def write_recording(recording, save_path, initial_sorting_fn, dtype='float32', verbose=True,
                        **job_kwargs):
        """ Convert and save the recording extractor to SHYBRID format

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved
        save_path: str
            Full path to desired target folder
        initial_sorting_fn: str
            Full path to the initial sorting csv file (can also be generated
            using write_sorting static method from the SHYBRIDSortingExtractor)
        dtype: dtype
            Type of the saved data. Default float32.
        **write_binary_kwargs: keyword arguments for write_to_binary_dat_format() function
        """
        assert HAVE_SBEX, SHYBRIDRecordingExtractor.installation_mesg
        save_path = Path(save_path)
        recording_name = 'recording.bin'
        probe_name = 'probe.prb'
        params_name = 'recording.yml'

        # location information has to be present in order for shybrid to
        # be able to operate on the recording
        if 'location' not in recording.get_shared_channel_property_names():
            raise GeometryNotLoadedError("Channel locations were not found")

        # write recording
        recording_fn = save_path / recording_name
        write_binary_recording(recording, files_path=recording_fn, dtype=dtype, verbose=verbose, **job_kwargs)
        recording.write_to_binary_dat_format(save_path=recording_fn,
                                             time_axis=0, dtype=dtype,
                                             **write_binary_kwargs)

        # write probe file
        probe_fn = os.path.join(save_path, PROBE_NAME)
        save_to_probe_file(recording, probe_fn)

        # create parameters file
        parameters = dict(clusters=initial_sorting_fn,
                          data=dict(dtype=dtype,
                                    fs=str(recording.get_sampling_frequency()),
                                    order='F',
                                    probe=probe_fn))

        # write parameters file
        parameters_fn = os.path.join(save_path, PARAMETERS_NAME)
        with open(parameters_fn, 'w') as fp:
            yaml.dump(parameters, fp)


class SHYBRIDRecordingSegment(BaseRecordingSegment):
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
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        recordings = self._diskreadmda.readChunk(i1=0, i2=start_frame, N1=self._diskreadmda.N1(),
                                                 N2=end_frame - start_frame)
        recordings = recordings[channel_indices, :].T
        return recordings


class SHYBRIDSortingExtractor(BaseSorting):
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


class SHYBRIDSortingSegment(BaseSortingSegment):
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
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        inds = np.where(
            (self._labels == unit_id) & (start_frame <= self._spike_times) & (self._spike_times < end_frame))
        return np.rint(self._spike_times[inds]).astype(int)


class GeometryNotLoadedError(Exception):
    """ Raised when the recording extractor has no associated channel locations
    """
    pass


params_template = \
    """clusters:
      csv: {initial_sorting_fn}
    data:
      dtype: {data_type}
      fs: {sampling_frequency}
      order: {byte_ordering}
      probe: {probe_fn}
    """
