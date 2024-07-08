from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseRecording, BaseSorting, BaseRecordingSegment, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from .matlabhelpers import MatlabHelper


class HDSortSortingExtractor(MatlabHelper, BaseSorting):
    """Load HDSort format data as a sorting extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to HDSort mat file.
    keep_good_only : bool, default: True
        Whether to only keep good units.

    Returns
    -------
    extractor : HDSortSortingExtractor
        The loaded data.
    """

    def __init__(self, file_path, keep_good_only=True):
        MatlabHelper.__init__(self, file_path)

        if not self._old_style_mat:
            _units = self._data["Units"]
            units = _parse_units(self._data, _units)

            # Extracting MutliElectrode field by field:
            _ME = self._data["MultiElectrode"]
            multi_electrode = dict((k, _ME.get(k)[()]) for k in _ME.keys())

            # Extracting sampling_frequency:
            sr = self._data["samplingRate"]
            sampling_frequency = float(_squeeze_ds(sr))

            # Remove noise units if necessary:
            if keep_good_only:
                units = [unit for unit in units if unit["ID"].flatten()[0].astype(int) % 1000 != 0]

            if "sortingInfo" in self._data.keys():
                info = self._data["sortingInfo"]
                start_frame = _squeeze_ds(info["startTimes"])
                self.start_frame = int(start_frame)
            else:
                self.start_frame = 0
        else:
            _units = self._getfield("Units").squeeze()
            fields = _units.dtype.fields.keys()
            units = []

            for unit in _units:
                unit_dict = {}
                for f in fields:
                    unit_dict[f] = unit[f]
                units.append(unit_dict)

            sr = self._getfield("samplingRate")
            sampling_frequency = float(_squeeze_ds(sr))

            _ME = self._data["MultiElectrode"]
            multi_electrode = dict((k, _ME[k][0][0].T) for k in _ME.dtype.fields.keys())

            # Remove noise units if necessary:
            if keep_good_only:
                units = [unit for unit in units if unit["ID"].flatten()[0].astype(int) % 1000 != 0]

            if "sortingInfo" in self._data.keys():
                info = self._getfield("sortingInfo")
                start_frame = _squeeze_ds(info["startTimes"])
                self.start_frame = int(start_frame)
            else:
                self.start_frame = 0

        self._units = units
        self._multi_electrode = multi_electrode

        unit_ids = []
        spiketrains = []
        for uc, unit in enumerate(units):
            unit_id = int(_squeeze_ds(unit["ID"]))
            spike_times = _squeeze(unit["spikeTrain"]).astype("int64") - self.start_frame
            unit_ids.append(unit_id)
            spiketrains.append(spike_times)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.add_sorting_segment(HDSortSortingSegment(unit_ids, spiketrains))

        # property
        templates = []
        templates_frames_cut_before = []
        for uc, unit in enumerate(units):
            if self._old_style_mat:
                template = unit["footprint"].T
            else:
                template = unit["footprint"]
            templates.append(template)
            templates_frames_cut_before.append(unit["cutLeft"].flatten())
        self.set_property("template", np.array(templates))
        self.set_property("template_frames_cut_before", np.array(templates_frames_cut_before))

        self._kwargs = {"file_path": str(Path(file_path).absolute()), "keep_good_only": keep_good_only}

        # TODO features
        # ~ for uc, unit in enumerate(units):
        # ~ unit_id = int(_squeeze_ds(unit["ID"]))
        # ~ self.set_unit_spike_features(unit_id, "amplitudes", _squeeze(unit["spikeAmplitudes"]))
        # ~ self.set_unit_spike_features(unit_id, "detection_channel", _squeeze(unit["detectionChannel"]).astype(np.int))
        # ~ idx = unit["detectionChannel"].astype(int) - 1
        # ~ spikePositions = np.vstack((_squeeze(multi_electrode["electrodePositions"][0][idx]),
        # ~ _squeeze(multi_electrode["electrodePositions"][1][idx]))).T
        # ~ self.set_unit_spike_features(unit_id, "positions", spikePositions)

    """
    @staticmethod
    def write_sorting(sorting, save_path, locations=None, noise_std_by_channel=None, start_frame=0):
        # First, find out how many channels there are
        if locations is not None:
            # write_locations must be a 2D numpy array with n_channels in first dim., (x,y) in second dim.
            n_channels = locations.shape[0]
        elif 'template' in sorting.get_shared_unit_property_names() or \
                'detection_channel' in sorting.get_shared_unit_property_names():
            # Without locations, check if there is a template to get the number of channels
            uid = int(sorting.get_unit_ids()[0])
            if "template" in sorting.get_unit_property_names(uid):
                template = sorting.get_unit_property(uid, "template")
                n_channels = template.shape[0]
            else:
                # If there is also no template, loop through all units and find max. detection_channel
                max_channel = 1
                for uid_ in sorting.get_unit_ids():
                    uid = int(uid_)
                    detection_channel = sorting.get_unit_spike_features(uid, "detection_channel")
                    max_channel = max([max_channel], np.append(detection_channel))
                n_channels = max_channel
        else:
            n_channels = 1

        # Now loop through all units and extract the data that we want to save:
        units = []
        for uid_ in sorting.get_unit_ids():
            uid = int(uid_)

            unit = {"ID": uid,
                    "spikeTrain": sorting.get_unit_spike_train(uid)}
            num_spikes = len(sorting.get_unit_spike_train(uid))

            if "amplitudes" in sorting.get_unit_spike_feature_names(uid):
                unit["spikeAmplitudes"] = sorting.get_unit_spike_features(uid, "amplitudes")
            else:
                # Save a spikeAmplitudes = 1
                unit["spikeAmplitudes"] = np.ones(num_spikes, np.double)

            if "detection_channel" in sorting.get_unit_spike_feature_names(uid):
                unit["detectionChannel"] = sorting.get_unit_spike_features(uid, "detection_channel")
            else:
                # Save a detectionChannel = 1
                unit["detectionChannel"] = np.ones(num_spikes, np.double)

            if "template" in sorting.get_unit_property_names(uid):
                unit["footprint"] = sorting.get_unit_property(uid, "template")
            else:
                # If this unit does not have a footprint, create an empty one:
                unit["footprint"] = np.zeros((3, n_channels), np.double)

            if "template_cut_left" in sorting.get_unit_property_names(uid):
                unit["cutLeft"] = sorting.get_unit_property(uid, "template_cut_left")
            else:
                unit["cutLeft"] = 1

            units.append(unit)

        # Save the electrode locations:
        if locations is None:
            # Create artificial locations if none are provided:
            x = np.zeros(n_channels, np.double)
            y = np.array(np.arange(n_channels), np.double)
            locations = np.vstack((x, y)).T

        multi_electrode = {"electrodePositions": locations, "electrodeNumbers": np.arange(n_channels)}

        if noise_std_by_channel is None:
            noise_std_by_channel = np.ones((1, n_channels))

        dict_to_save = {"Units": units,
                        "MultiElectrode": multi_electrode,
                        "noiseStd": noise_std_by_channel,
                        "samplingRate": sorting._sampling_frequency}

        # Save Units and MultiElectrode to .mat file:
        MATSortingExtractor.write_dict_to_mat(save_path, dict_to_save, version="7.3")
    """


class HDSortSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spiketrains):
        BaseSortingSegment.__init__(self)
        self._unit_ids = list(unit_ids)
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        unit_index = self._unit_ids.index(unit_id)
        times = self._spiketrains[unit_index]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


# For .mat v7.3: Function to extract all fields of a struct-array:
def _parse_units(file, _units):
    import h5py

    t_units = {}
    if isinstance(_units, h5py.Group):
        for name in _units.keys():
            value = _units[name]
            dict_val = []
            for val in value:
                if isinstance(file[val[0]], h5py.Dataset):
                    dict_val.append(file[val[0]][()])
                    t_units[name] = dict_val
                else:
                    break
        out = [dict(zip(t_units, col)) for col in zip(*t_units.values())]
    else:
        out = []
        for unit in _units:
            group = file[unit[()][0]]
            unit_dict = {}
            for k in group.keys():
                unit_dict[k] = group[k][()]
            out.append(unit_dict)

    return out


def _squeeze_ds(ds):
    while not isinstance(ds, (int, float, np.integer)):
        ds = ds[0]
    return ds


def _squeeze(arr):
    shape = arr.shape
    if len(shape) == 2:
        if shape[0] == 1:
            arr = arr[0]
        elif shape[1] == 1:
            arr = arr[:, 0]
    return arr


read_hdsort = define_function_from_class(source_class=HDSortSortingExtractor, name="read_hdsort")
