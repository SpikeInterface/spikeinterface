from pathlib import Path
import json

from copy import deepcopy
import shutil

import numpy as np

from probeinterface import read_probeinterface, write_probeinterface

from spikeinterface.core import BaseRecording
from .binaryrecordingextractor import BinaryRecordingExtractor
from .core_tools import (
    define_function_from_class,
    make_paths_absolute,
    load_properties_from_binary_folder,
    save_properties_to_binary_folder,
    save_extractor_provenance,
    save_annotations_to_folder,
    load_annotations_from_folder,
)


class BinaryFolderRecording(BinaryRecordingExtractor):
    """
    BinaryFolderRecording is an internal format used in spikeinterface.
    It is a BinaryRecordingExtractor + metadata contained in a folder.

    It is created with the function: `recording.save(format="binary", folder="/myfolder")`

    Parameters
    ----------
    folder_path : str or Path

    Returns
    -------
    recording : BinaryFolderRecording
        The recording
    """

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "binary.json", "r") as f:
            d = json.load(f)

        if not d["class"].endswith(".BinaryRecordingExtractor"):
            raise ValueError("This folder is not a binary spikeinterface folder")

        assert d["relative_paths"]

        d = make_paths_absolute(d, folder_path)

        BinaryRecordingExtractor.__init__(self, **d["kwargs"])

        # Load properties
        load_properties_from_binary_folder(folder_path / "properties", self)
        load_annotations_from_folder(folder_path, self)

        # Load the probegroup
        probe_file = folder_path / "probegroup.json"
        # In spikeinterface version < 0.105.0, the probegroup was saved in a file called probe.json
        legacy_probe_file = folder_path / "probe.json"
        probegroup = None
        if probe_file.is_file():
            # This is the new version: the probegroup is already ordered correctly
            probegroup = read_probeinterface(probe_file)
        elif legacy_probe_file.is_file():
            probegroup = read_probeinterface(legacy_probe_file)
            order = np.argsort(probegroup.to_numpy(complete=True)["device_channel_indices"])
            if not np.array_equal(order, np.arange(len(order))):
                # In spikeinterface version < 0.105.0, the order was saved in the contact vector, but not
                # in the probegroup. We need to check if the order is correct and if not, we need to reorder
                # the probegroup to match the channel ids.
                probegroup = probegroup.get_slice(order)

            # In some older SI versions, before #4300, the probe annotations were
            # saved to the recording annotations as `probes_info`. If this is the
            # case, we can copy the annotations to the probegroup and delete the
            # `probes_info` from the recording annotations.
            si_folder_json = folder_path / "si_folder.json"
            if si_folder_json.is_file():
                with open(si_folder_json, "r") as f:
                    si_folder_dict = json.load(f)
                if "annotations" in si_folder_dict:
                    si_annotations = si_folder_dict["annotations"]
                    if "probes_info" in si_annotations:
                        probes_info = si_annotations.pop("probes_info")
                        for probe, probe_info in zip(probegroup.probes, probes_info):
                            probe.annotations.update(probe_info)

        if probegroup is not None:
            self._probegroup = probegroup

        # Load time vectors if any
        for segment_index, rs in enumerate(self.segments):
            time_file = folder_path / f"times_cached_seg{segment_index}.npy"
            if time_file.is_file():
                rs._time_vector = np.load(time_file, mmap_mode="r")

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._bin_kwargs = d["kwargs"]
        if "num_channels" not in self._bin_kwargs:
            assert "num_chan" in self._bin_kwargs, "Cannot find num_channels or num_chan in binary.json"
            self._bin_kwargs["num_channels"] = self._bin_kwargs["num_chan"]

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._bin_kwargs["file_paths"],
            dtype=np.dtype(self._bin_kwargs["dtype"]),
            num_channels=self._bin_kwargs["num_channels"],
            time_axis=self._bin_kwargs["time_axis"],
            file_offset=self._bin_kwargs["file_offset"],
        )
        return d

    @staticmethod
    def write_recording(
        recording: BaseRecording,
        folder_path: str | Path,
        verbose: bool = False,
        overwrite: bool = False,
        dtype=None,
        **job_kwargs,
    ):
        from .time_series_tools import write_binary
        from .binaryrecordingextractor import BinaryRecordingExtractor
        from .binaryfolder import BinaryFolderRecording

        folder_path = Path(folder_path)
        if folder_path.is_dir():
            if not overwrite:
                raise FileExistsError(f"Folder {folder_path} already exists. Use overwrite=True to overwrite it.")
            else:
                shutil.rmtree(folder_path)
        folder_path.mkdir(exist_ok=False, parents=True)

        file_paths = [folder_path / f"traces_cached_seg{i}.raw" for i in range(recording.get_num_segments())]
        if dtype is None:
            dtype = recording.get_dtype()
        # Check if there are any time vectors
        t_starts = recording.get_segment_t_starts()
        if recording.has_any_time_vector():
            file_timestamps_paths = [
                folder_path / f"times_cached_seg{i}.raw" for i in range(recording.get_num_segments())
            ]
        else:
            file_timestamps_paths = None

        write_binary(
            recording,
            file_paths=file_paths,
            file_timestamps_paths=file_timestamps_paths,
            dtype=dtype,
            verbose=verbose,
            **job_kwargs,
        )

        save_properties_to_binary_folder(folder_path / "properties", recording)
        save_extractor_provenance(folder_path, recording)
        # new in version 0.105.0, before that annotations were handle by "si_folder.json" file
        save_annotations_to_folder(folder_path, recording)

        if recording.has_probe():
            probegroup = recording.get_probegroup()
            write_probeinterface(folder_path / "probegroup.json", probegroup)

        # This is created so it can be saved as json because the `BinaryFolderRecording` requires it loading
        # See the __init__
        binary_rec = BinaryRecordingExtractor(
            file_paths=file_paths,
            file_timestamps_paths=file_timestamps_paths,
            sampling_frequency=recording.get_sampling_frequency(),
            num_channels=recording.get_num_channels(),
            dtype=dtype,
            t_starts=t_starts,
            channel_ids=recording.get_channel_ids(),
            time_axis=0,
            file_offset=0,
            is_filtered=recording.is_filtered(),
            gain_to_uV=recording.get_channel_gains(),
            offset_to_uV=recording.get_channel_offsets(),
        )
        binary_rec.dump(folder_path / "binary.json", relative_to=folder_path)

        # Create the si_folder file to make the load() easier until version 0.105.0
        # All properties, annotations, and probe information are already saved in the folder,
        # so we don't need to include them in the si_folder.json
        cached = BinaryFolderRecording(folder_path=folder_path)
        si_folder_path = folder_path / f"si_folder.json"
        cached.dump_to_json(
            file_path=si_folder_path,
            relative_to=folder_path,
            include_properties=False,
            include_annotations=False,
            include_extra_metadata=False,
        )

        return cached


read_binary_folder = define_function_from_class(source_class=BinaryFolderRecording, name="read_binary_folder")
