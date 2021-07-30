from pathlib import Path
import numpy as np
from typing import Union, List

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment

try:
    import pandas as pd
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries, FilteredEphys, LFP
    from pynwb.ecephys import ElectrodeGroup
    from hdmf.data_utils import DataChunkIterator
    from hdmf.backends.hdf5.h5_utils import H5DataIO

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False

PathType = Union[str, Path, None]


def check_nwb_install():
    assert HAVE_NWB, NwbRecordingExtractor.installation_mesg


class NwbRecordingExtractor(BaseRecording):
    """Primary class for interfacing between NWBFiles and RecordingExtractors."""

    extractor_name = 'NwbRecording'
    has_default_locations = True
    has_unscaled = False
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path: PathType, electrical_series_name: str = None):
        """
        Load an NWBFile as a RecordingExtractor.

        Parameters
        ----------
        file_path: path to NWB file
        electrical_series_name: str, optional
        """
        assert HAVE_NWB, self.installation_mesg
        self._file_path = str(file_path)
        with NWBHDF5IO(self._file_path, 'r') as io:
            nwbfile = io.read()
            if electrical_series_name is not None:
                electrical_series_name = electrical_series_name
            else:
                a_names = list(nwbfile.acquisition)
                if len(a_names) > 1:
                    raise ValueError("More than one acquisition found! You must specify 'electrical_series_name'.")
                if len(a_names) == 0:
                    raise ValueError("No acquisitions found in the .nwb file.")
                electrical_series_name = a_names[0]
            es = nwbfile.acquisition[electrical_series_name]
            if hasattr(es, 'timestamps') and es.timestamps:
                sampling_frequency = 1. / np.median(np.diff(es.timestamps))
                recording_start_time = es.timestamps[0]
            else:
                sampling_frequency = es.rate
                if hasattr(es, 'starting_time'):
                    recording_start_time = es.starting_time
                else:
                    recording_start_time = 0.

            num_frames = int(es.data.shape[0])
            num_channels = len(es.electrodes.data)

            # Channels gains - for RecordingExtractor, these are values to cast traces to uV
            if es.channel_conversion is not None:
                gains = es.conversion * es.channel_conversion[:] * 1e6
            else:
                gains = es.conversion * np.ones(num_channels) * 1e6
            # Extractors channel groups must be integers, but Nwb electrodes group_name can be strings
            if 'group_name' in nwbfile.electrodes.colnames:
                unique_grp_names = list(np.unique(nwbfile.electrodes['group_name'][:]))

            # Fill channel properties dictionary from electrodes table
            channel_ids = [es.electrodes.table.id[x] for x in es.electrodes.data]

            dtype = es.data.dtype

            BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
            recording_segment = NwbRecordingSegment(path=self._file_path, electrical_series_name=electrical_series_name,
                                                    num_frames=num_frames)
            self.add_recording_segment(recording_segment)

            # If gains are not 1, set has_scaled to True
            if np.any(gains != 1):
                self.set_channel_gains(gains)

            # Add properties
            properties = dict()
            for es_ind, (channel_id, electrode_table_index) in enumerate(zip(channel_ids, es.electrodes.data)):
                if 'rel_x' in nwbfile.electrodes:
                    if 'location' not in properties:
                        properties['location'] = np.zeros((self.get_num_channels(), 2), dtype=float)
                    properties['location'][es_ind, 0] = nwbfile.electrodes['rel_x'][electrode_table_index]
                    if 'rel_y' in nwbfile.electrodes:
                        properties['location'][es_ind, 1] = nwbfile.electrodes['rel_x'][electrode_table_index]

                for col in nwbfile.electrodes.colnames:
                    if isinstance(nwbfile.electrodes[col][electrode_table_index], ElectrodeGroup):
                        continue
                    elif col == 'group_name':
                        group = unique_grp_names.index(nwbfile.electrodes[col][electrode_table_index])
                        if 'group' not in properties:
                            properties['group'] = np.zeros(self.get_num_channels(), dtype=type(group))
                        properties['group'][es_ind] = group
                    elif col == 'location':
                        brain_area = nwbfile.electrodes[col][electrode_table_index]
                        if 'brain_area' not in properties:
                            properties['brain_area'] = np.zeros(self.get_num_channels(), dtype=type(brain_area))
                        properties['brain_area'][es_ind] = brain_area
                    elif col == 'offset':
                        offset = nwbfile.electrodes[col][electrode_table_index]
                        if 'offset' not in properties:
                            properties['offset'] = np.zeros(self.get_num_channels(), dtype=type(offset))
                        properties['offset'][es_ind] = offset
                    elif col in ['x', 'y', 'z', 'rel_x', 'rel_y']:
                        continue
                    else:
                        val = nwbfile.electrodes[col][electrode_table_index]
                        if col not in properties:
                            properties[col] = np.zeros(self.get_num_channels(), dtype=type(val))
                        properties[col][es_ind] = val

            for prop_name, values in properties.items():
                if prop_name == "location":
                    self.set_dummy_probe_from_locations(values)
                elif prop_name == "group":
                    if np.isscalar(val):
                        groups = [val] * len(channel_ids)
                    else:
                        groups = val
                    self.set_channel_groups(groups)
                else:
                    self.set_property(prop_name, values)

            self._kwargs = {'file_path': str(Path(file_path).absolute()),
                            'electrical_series_name': electrical_series_name}


class NwbRecordingSegment(BaseRecordingSegment):
    def __init__(self, path: PathType, electrical_series_name, num_frames):
        BaseRecordingSegment.__init__(self)
        self._path = path
        self._electrical_series_name = electrical_series_name
        self._num_samples = num_frames

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            es = nwbfile.acquisition[self._electrical_series_name]

            if isinstance(channel_indices, slice):
                traces = es.data[start_frame:end_frame, channel_indices]
            else:
                # channel_indices is np.ndarray
                if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                    # get around h5py constraint that it does not allow datasets
                    # to be indexed out of order
                    sorted_channel_indices = np.sort(channel_indices)
                    resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                    recordings = es.data[start_frame:end_frame, sorted_channel_indices]
                    traces = recordings[:, resorted_indices]
                else:
                    traces = es.data[start_frame:end_frame, channel_indices]

        return traces


class NwbSortingExtractor(BaseSorting):
    extractor_name = 'NwbSorting'
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path, electrical_series_name: str = None, sampling_frequency: float = None):
        """
        Parameters
        ----------
        file_path: path to NWB file
        electrical_series_name: str with pynwb.ecephys.ElectricalSeries object name
        sampling_frequency: float
        """
        assert self.installed, self.installation_mesg
        self._file_path = str(file_path)
        with NWBHDF5IO(self._file_path, 'r') as io:
            nwbfile = io.read()
            if sampling_frequency is None:
                # defines the electrical series from where the sorting came from
                # important to know the sampling_frequency
                if electrical_series_name is None:
                    if len(nwbfile.acquisition) > 1:
                        raise Exception('More than one acquisition found. You must specify electrical_series_name.')
                    if len(nwbfile.acquisition) == 0:
                        raise Exception("No acquisitions found in the .nwb file from which to read sampling frequency. \
                                         Please, specify 'sampling_frequency' parameter.")
                    es = list(nwbfile.acquisition.values())[0]
                else:
                    es = electrical_series_name
                # get rate
                if es.rate is not None:
                    sampling_frequency = es.rate
                else:
                    sampling_frequency = 1 / (es.timestamps[1] - es.timestamps[0])

            assert sampling_frequency is not None, "Couldn't load sampling frequency. Please provide it with the " \
                                                   "'sampling_frequency' argument"

            # get all units ids
            units_ids = list(nwbfile.units.id[:])

            # store units properties and spike features to dictionaries
            all_pr_ft = list(nwbfile.units.colnames)
            all_names = [i.name for i in nwbfile.units.columns]
            for item in all_pr_ft:
                if item == 'spike_times':
                    continue
                # test if item is a unit_property or a spike_feature
                if item + '_index' in all_names:  # if it has index, it is a spike_feature
                    pass
                else:  # if it is unit_property
                    properties = dict()
                    for u_id in units_ids:
                        ind = list(units_ids).index(u_id)
                        if isinstance(nwbfile.units[item][ind], pd.DataFrame):
                            prop_value = nwbfile.units[item][ind].index[0]
                        else:
                            prop_value = nwbfile.units[item][ind]

                        if item not in properties:
                            properties[item] = np.zeros(len(units_ids), dtype=type(prop_value))

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, units_ids=units_ids)
        sorting_segment = NwbSortingSegment(path=self._file_path, sampling_frequency=sampling_frequency)
        self.add_sorting_segment(sorting_segment)

        for prop_name, values in properties.items():
            self.set_property(prop_name, values)

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'electrical_series_name': electrical_series_name,
                        'sampling_frequency': sampling_frequency}


class NwbSortingSegment(BaseSortingSegment):
    def __init__(self, path, sampling_frequency):
        BaseSortingSegment.__init__(self)
        self._path = path
        self._sampling_frequency = sampling_frequency

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            # chosen unit and interval
            times = nwbfile.units['spike_times'][list(nwbfile.units.id[:]).index(unit_id)][:]
            # spike times are measured in samples
            # TODO if present, use times from file
            frames = np.round(times * self._sampling_frequency()).astype('int64')
        return frames[(frames > start_frame) & (frames < end_frame)]


def read_nwb_recording(*args, **kwargs):
    recording = NwbRecordingExtractor(*args, **kwargs)
    return recording


def read_nwb_sorting(*args, **kwargs):
    sorting = NwbSortingExtractor(*args, **kwargs)
    return sorting


def read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None):
    outputs = ()
    if load_recording:
        rec = read_nwb_recording(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (rec,)
    if load_sorting:
        sorting = read_nwb_sorting(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (sorting,)
    return outputs
