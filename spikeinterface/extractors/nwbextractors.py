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


def get_electrical_series(nwbfile, electrical_series_name):
    if electrical_series_name is not None:
        es_dict = {i.name: i for i in nwbfile.all_children() if isinstance(i, ElectricalSeries)}
        assert electrical_series_name in es_dict, 'electrical series name not present in nwbfile'
        es = es_dict[electrical_series_name]
    else:
        es_list = []
        for name, series in nwbfile.acquisition.items():
            if isinstance(series, ElectricalSeries):
                es_list.append(series)
        if len(es_list) > 1:
            raise ValueError("More than one acquisition found! You must specify 'electrical_series_name'.")
        if len(es_list) == 0:
            raise ValueError("No acquisitions found in the .nwb file.")
        es = es_list[0]
    return es


class NwbRecordingExtractor(BaseRecording):
    """
    Load an NWBFile as a RecordingExtractor.

    Parameters
    ----------
    file_path: str or Path
        Path to NWB file
    electrical_series_name: str, optional
        The name of the ElectricalSeries (if multiple ElectricalSeries are present)
    load_time_vector: bool
        If True, the time vector is loaded to the recording object (default False)
    samples_for_rate_estimation: int
        If 'rate' is not specified in the ElectricalSeries, number of timestamps samples to use
        to estimate the rate (default 100000)

    Returns
    -------
    recording: NwbRecordingExtractor
        The recording extractor for the NWB file
    """
    extractor_name = 'NwbRecording'
    has_default_locations = True
    has_unscaled = False
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path: PathType, electrical_series_name: str = None, load_time_vector: bool = False,
                 samples_for_rate_estimation: int = 100000):
        check_nwb_install()
        self._file_path = str(file_path)
        self._electrical_series_name = electrical_series_name

        self.io = NWBHDF5IO(self._file_path, mode='r', load_namespaces=True)
        self._nwbfile = self.io.read()
        self._es = get_electrical_series(
            self._nwbfile, self._electrical_series_name)

        sampling_frequency = None
        if hasattr(self._es, 'rate'):
            sampling_frequency = self._es.rate

        if hasattr(self._es, 'starting_time'):
            t_start = self._es.starting_time
        else:
            t_start = None

        timestamps = None
        if hasattr(self._es, 'timestamps'):
            if self._es.timestamps is not None:
                timestamps = self._es.timestamps
                t_start = self._es.timestamps[0]

        # if rate is unknown, estimate from timestamps
        if sampling_frequency is None:
            assert timestamps is not None, "Could not find rate information as both 'rate' and "\
                                           "'timestamps' are missing from the file. "\
                                           "Use the 'sampling_frequency' argument."
            sampling_frequency = 1. / np.median(np.diff(timestamps[:samples_for_rate_estimation]))

        if load_time_vector and timestamps is not None:
            times_kwargs = dict(time_vector=self._es.timestamps)
        else:
            times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=t_start)

        num_frames = int(self._es.data.shape[0])
        num_channels = len(self._es.electrodes.data)

        # Channels gains - for RecordingExtractor, these are values to cast traces to uV
        if self._es.channel_conversion is not None:
            gains = self._es.conversion * self._es.channel_conversion[:] * 1e6
        else:
            gains = self._es.conversion * np.ones(num_channels) * 1e6
        # Extractors channel groups must be integers, but Nwb electrodes group_name can be strings
        if 'group_name' in self._nwbfile.electrodes.colnames:
            unique_grp_names = list(np.unique(self._nwbfile.electrodes['group_name'][:]))

        # Fill channel properties dictionary from electrodes table
        channel_ids = [self._es.electrodes.table.id[x]
                       for x in self._es.electrodes.data]

        dtype = self._es.data.dtype

        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = NwbRecordingSegment(nwbfile=self._nwbfile,
                                                electrical_series_name=self._electrical_series_name,
                                                num_frames=num_frames, times_kwargs=times_kwargs)
        self.add_recording_segment(recording_segment)

        # If gains are not 1, set has_scaled to True
        if np.any(gains != 1):
            self.set_channel_gains(gains)

        # Add properties
        properties = dict()
        for es_ind, (channel_id, electrode_table_index) in enumerate(zip(channel_ids, self._es.electrodes.data)):
            if 'rel_x' in self._nwbfile.electrodes:
                if 'location' not in properties:
                    properties['location'] = np.zeros((self.get_num_channels(), 2), dtype=float)
                properties['location'][es_ind, 0] = self._nwbfile.electrodes['rel_x'][electrode_table_index]
                if 'rel_y' in self._nwbfile.electrodes:
                    properties['location'][es_ind, 1] = self._nwbfile.electrodes['rel_y'][electrode_table_index]

            for col in self._nwbfile.electrodes.colnames:
                if isinstance(self._nwbfile.electrodes[col][electrode_table_index], ElectrodeGroup):
                    continue
                elif col == 'group_name':
                    group = unique_grp_names.index(
                        self._nwbfile.electrodes[col][electrode_table_index])
                    if 'group' not in properties:
                        properties['group'] = np.zeros(self.get_num_channels(), dtype=type(group))
                    properties['group'][es_ind] = group
                elif col == 'location':
                    brain_area = self._nwbfile.electrodes[col][electrode_table_index]
                    if 'brain_area' not in properties:
                        properties['brain_area'] = np.zeros(self.get_num_channels(), dtype=type(brain_area))
                    properties['brain_area'][es_ind] = brain_area
                elif col == 'offset':
                    offset = self._nwbfile.electrodes[col][electrode_table_index]
                    if 'offset' not in properties:
                        properties['offset'] = np.zeros(self.get_num_channels(), dtype=type(offset))
                    properties['offset'][es_ind] = offset
                elif col in ['x', 'y', 'z', 'rel_x', 'rel_y']:
                    continue
                else:
                    val = self._nwbfile.electrodes[col][electrode_table_index]
                    if col not in properties:
                        properties[col] = np.zeros(self.get_num_channels(), dtype=type(val))
                    properties[col][es_ind] = val

        for prop_name, values in properties.items():
            if prop_name == "location":
                self.set_dummy_probe_from_locations(values)
            elif prop_name == "group":
                if np.isscalar(values):
                    groups = [values] * len(channel_ids)
                else:
                    groups = values
                self.set_channel_groups(groups)
            else:
                self.set_property(prop_name, values)

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'electrical_series_name': self._electrical_series_name,
                        'load_time_vector': load_time_vector,
                        'samples_for_rate_estimation': samples_for_rate_estimation}


class NwbRecordingSegment(BaseRecordingSegment):
    def __init__(self, nwbfile, electrical_series_name, num_frames, times_kwargs):
        BaseRecordingSegment.__init__(self, **times_kwargs)
        self._nwbfile = nwbfile
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

        es = get_electrical_series(self._nwbfile, self._electrical_series_name)

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
    """
    Load an NWBFile as a SortingExtractor.

    Parameters
    ----------
    file_path: str or Path
        Path to NWB file
    electrical_series_name: str, optional
        The name of the ElectricalSeries (if multiple ElectricalSeries are present)
    sampling_frequency: float, optional
        The sampling frequency in Hz (required if no ElectricalSeries is available)
    samples_for_rate_estimation: int, optional
        If 'rate' is not specified in the ElectricalSeries, number of timestamps samples to use
        to estimate the rate (default 100000)

    Returns
    -------
    sorting: NwbSortingExtractor
        The sorting extractor for the NWB file
    """
    extractor_name = 'NwbSorting'
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path: PathType, electrical_series_name: str = None, sampling_frequency: float = None,
                 samples_for_rate_estimation: int = 100000):
        check_nwb_install()
        self._file_path = str(file_path)
        self._electrical_series_name = electrical_series_name

        self.io = NWBHDF5IO(self._file_path, mode='r', load_namespaces=True)
        self._nwbfile = self.io.read()
        timestamps = None
        if sampling_frequency is None:
            # defines the electrical series from where the sorting came from
            # important to know the sampling_frequency
            self._es = get_electrical_series(self._nwbfile, self._electrical_series_name)
            # get rate
            if self._es.rate is not None:
                sampling_frequency = self._es.rate
            else:
                if hasattr(self._es, "timestamps"):
                    if self._es.timestamps is not None:
                        timestamps = self._es.timestamps
                        sampling_frequency = 1 / np.median(np.diff(timestamps[samples_for_rate_estimation]))

        assert sampling_frequency is not None, "Couldn't load sampling frequency. Please provide it with the " \
                                               "'sampling_frequency' argument"

        # get all units ids
        units_ids = list(self._nwbfile.units.id[:])

        # store units properties and spike features to dictionaries
        properties = dict()

        for column in list(self._nwbfile.units.colnames):
            if column == 'spike_times':
                continue
            # if it is unit_property
            property_values = self._nwbfile.units[column][:]

            # only load columns with same shape for all units
            if np.all(p.shape == property_values[0].shape for p in property_values):
                properties[column] = property_values
            else:
                print(f"Skipping {column} because of unequal shapes across units")

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=units_ids)
        sorting_segment = NwbSortingSegment(nwbfile=self._nwbfile, sampling_frequency=sampling_frequency,
                                            timestamps=timestamps)
        self.add_sorting_segment(sorting_segment)

        for prop_name, values in properties.items():
            self.set_property(prop_name, np.array(values))

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'electrical_series_name': self._electrical_series_name,
                        'sampling_frequency': sampling_frequency,
                        'samples_for_rate_estimation': samples_for_rate_estimation}


class NwbSortingSegment(BaseSortingSegment):
    def __init__(self, nwbfile, sampling_frequency, timestamps):
        BaseSortingSegment.__init__(self)
        self._nwbfile = nwbfile
        self._sampling_frequency = sampling_frequency
        self._timestamps = timestamps

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
        times = self._nwbfile.units['spike_times'][list(self._nwbfile.units.id[:]).index(unit_id)][:]

        if self._timestamps is not None:
            frames = np.searchsorted(times, self.timestamps).astype('int64')
        else:
            frames = np.round(times * self._sampling_frequency).astype('int64')
        return frames[(frames > start_frame) & (frames < end_frame)]


def read_nwb_recording(*args, **kwargs):
    recording = NwbRecordingExtractor(*args, **kwargs)
    return recording


read_nwb_recording.__doc__ = NwbRecordingExtractor.__doc__


def read_nwb_sorting(*args, **kwargs):
    sorting = NwbSortingExtractor(*args, **kwargs)
    return sorting


read_nwb_sorting.__doc__ = NwbSortingExtractor.__doc__


def read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None):
    """Reads NWB file into SpikeInterface extractors

    Parameters
    ----------
    file_path: str or Path
            Path to NWB file
    load_recording : bool, optional
        If True, the recording object is loaded (default True)
    load_sorting : bool, optional
        If True, the recording object is loaded (default False)
    electrical_series_name: str, optional
        The name of the ElectricalSeries (if multiple ElectricalSeries are present)

    Returns
    -------
    extractors: extractor or tuple
        Single RecordingExtractor/SortingExtractor or tuple with both
        (depending on 'load_recording'/'load_sorting') arguments
    """
    outputs = ()
    if load_recording:
        rec = read_nwb_recording(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (rec,)
    if load_sorting:
        sorting = read_nwb_sorting(file_path, electrical_series_name=electrical_series_name)
        outputs = outputs + (sorting,)
    return outputs
