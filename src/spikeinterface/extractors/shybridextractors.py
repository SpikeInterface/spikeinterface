from __future__ import annotations

from pathlib import Path

import numpy as np

import probeinterface

from spikeinterface.core import (
    BinaryRecordingExtractor,
    BaseRecordingSegment,
    BaseSorting,
    BaseSortingSegment,
    write_binary_recording,
)
from spikeinterface.core.core_tools import define_function_from_class


class SHYBRIDRecordingExtractor(BinaryRecordingExtractor):
    """Load SHYBRID format data as a recording extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the SHYBRID file.

    Returns
    -------
    extractor : SHYBRIDRecordingExtractor
        Loaded data.
    """

    installation_mesg = (
        "To use the SHYBRID extractors, install SHYBRID and pyyaml: " "\n\n pip install shybrid pyyaml\n\n"
    )

    def __init__(self, file_path):
        try:
            import hybridizer.io as sbio
            import hybridizer.probes as sbprb
            import yaml
        except ImportError:
            raise ImportError(self.installation_mesg)

        # load params file related to the given shybrid recording
        assert Path(file_path).suffix in [".yml", ".yaml"], "The 'file_path' should be a yaml file!"
        params = sbio.get_params(file_path)["data"]
        file_path = Path(file_path)

        # create a shybrid probe object
        probe = sbprb.Probe(params["probe"])
        nb_channels = probe.total_nb_channels

        # translate the byte ordering
        byte_order = params["order"]
        if byte_order == "C":
            time_axis = 1
        elif byte_order == "F":
            time_axis = 0

        bin_file = file_path.parent / f"{file_path.stem}.bin"

        # piggyback on binary data recording extractor
        BinaryRecordingExtractor.__init__(
            self,
            file_paths=bin_file,
            sampling_frequency=float(params["fs"]),
            num_channels=nb_channels,
            dtype=params["dtype"],
            time_axis=time_axis,
        )

        # load probe file
        probegroup = probeinterface.read_prb(params["probe"])
        self.set_probegroup(probegroup, in_place=True)
        self._kwargs = {"file_path": str(Path(file_path).absolute())}
        self.extra_requirements.extend(["hybridizer", "pyyaml"])

    @staticmethod
    def write_recording(recording, save_path, initial_sorting_fn, dtype="float32", **job_kwargs):
        """Convert and save the recording extractor to SHYBRID format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved
        save_path: str
            Full path to desired target folder
        initial_sorting_fn: str
            Full path to the initial sorting csv file (can also be generated
            using write_sorting static method from the SHYBRIDSortingExtractor)
        dtype: dtype, default: float32
            Type of the saved data
        **write_binary_kwargs: keyword arguments for write_to_binary_dat_format() function
        """
        try:
            import hybridizer.io as sbio
            import hybridizer.probes as sbprb
            import yaml
        except ImportError:
            raise ImportError(SHYBRIDRecordingExtractor.installation_mesg)

        assert recording.get_num_segments() == 1, "SHYBRID can only write single segment recordings"
        save_path = Path(save_path)
        recording_name = "recording.bin"
        probe_name = "probe.prb"
        params_name = "recording.yml"

        # location information has to be present in order for shybrid to
        # be able to operate on the recording
        if recording.get_channel_locations() is None:
            raise GeometryNotLoadedError("Channel locations were not found")

        # write recording
        recording_fn = (save_path / recording_name).absolute()
        write_binary_recording(recording, file_paths=recording_fn, dtype=dtype, **job_kwargs)

        # write probe file
        probe_fn = (save_path / probe_name).absolute()
        probegroup = recording.get_probegroup()
        probeinterface.write_prb(probe_fn, probegroup, total_nb_channels=recording.get_num_channels())

        # create parameters file
        parameters = dict(
            clusters=initial_sorting_fn,
            data=dict(dtype=dtype, fs=str(recording.get_sampling_frequency()), order="F", probe=str(probe_fn)),
        )

        # write parameters file
        parameters_fn = (save_path / params_name).absolute()
        with parameters_fn.open("w") as fp:
            yaml.dump(parameters, fp)


class SHYBRIDSortingExtractor(BaseSorting):
    """Load SHYBRID format data as a sorting extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the SHYBRID file.
    sampling_frequency : int
        The sampling frequency.
    delimiter : str
        The delimiter to use for loading the file.

    Returns
    -------
    extractor : SHYBRIDSortingExtractor
        Loaded data.
    """

    installation_mesg = "To use the SHYBRID extractors, install SHYBRID: \n\n pip install shybrid\n\n"

    def __init__(self, file_path, sampling_frequency, delimiter=","):
        try:
            import hybridizer.io as sbio
            import hybridizer.probes as sbprb
        except ImportError:
            raise ImportError(self.installation_mesg)

        assert Path(file_path).suffix == ".csv", "The 'file_path' should be a csv file!"

        if Path(file_path).is_file():
            spike_clusters = sbio.SpikeClusters()
            spike_clusters.fromCSV(str(file_path), None, delimiter=delimiter)
        else:
            raise FileNotFoundError(f"The ground truth file {file_path} could not be found")

        BaseSorting.__init__(self, unit_ids=spike_clusters.keys(), sampling_frequency=sampling_frequency)

        sorting_segment = SHYBRIDSortingSegment(spike_clusters)
        self.add_sorting_segment(sorting_segment)

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
            "delimiter": delimiter,
        }
        self.extra_requirements.append("hybridizer")

    @staticmethod
    def write_sorting(sorting, save_path):
        """Convert and save the sorting extractor to SHYBRID CSV format.

        Parameters
        ----------
        sorting : SortingExtractor
            The sorting extractor to be converted and saved.
        save_path : str
            Full path to the desired target folder.
        """
        try:
            import hybridizer.io as sbio
            import hybridizer.probes as sbprb
        except ImportError:
            raise ImportError(SHYBRIDSortingExtractor.installation_mesg)

        assert sorting.get_num_segments() == 1, "SHYBRID can only write single segment sortings"
        save_path = Path(save_path)

        dump = np.empty((0, 2))

        for unit_id in sorting.get_unit_ids():
            spikes = sorting.get_unit_spike_train(unit_id)[:, np.newaxis]
            expanded_id = (np.ones(spikes.size) * unit_id)[:, np.newaxis]
            tmp_concat = np.concatenate((expanded_id, spikes), axis=1)

            dump = np.concatenate((dump, tmp_concat), axis=0)

        save_path.mkdir(exist_ok=True, parents=True)
        sorting_fn = save_path / "initial_sorting.csv"
        np.savetxt(sorting_fn, dump, delimiter=",", fmt="%i")


class SHYBRIDSortingSegment(BaseSortingSegment):
    def __init__(self, spike_clusters):
        self._spike_clusters = spike_clusters
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame,
        end_frame,
    ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        train = self._spike_clusters[unit_id].get_actual_spike_train().spikes
        idxs = np.where((start_frame <= train) & (train < end_frame))
        return train[idxs]


read_shybrid_recording = define_function_from_class(
    source_class=SHYBRIDRecordingExtractor, name="read_shybrid_recording"
)
read_shybrid_sorting = define_function_from_class(source_class=SHYBRIDSortingExtractor, name="read_shybrid_sorting")


class GeometryNotLoadedError(Exception):
    """Raised when the recording extractor has no associated channel locations."""

    pass


params_template = """clusters:
      csv: {initial_sorting_fn}
    data:
      dtype: {data_type}
      fs: {sampling_frequency}
      order: {byte_ordering}
      probe: {probe_fn}
    """
