from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class XClustSortingExtractor(BaseSorting):
    """Load XClust sorting solution as a sorting extractor.

    XClust is a legacy spike sorting tool from the McNaughton lab. Each `.CEL`
    file is ASCII with a header (``%%BEGINHEADER`` / ``%%ENDHEADER``) followed
    by whitespace-separated tabular data containing spike times.

    Parameters
    ----------
    folder_path : str or Path or None, default: None
        Path to folder containing `.CEL` files. Mutually exclusive with
        ``file_path_list``.
    file_path_list : list of str or Path or None, default: None
        Explicit list of `.CEL` file paths. Mutually exclusive with
        ``folder_path``.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    extractor : XClustSortingExtractor
        Loaded data.
    """

    def __init__(
        self,
        folder_path: str | Path | None = None,
        *,
        file_path_list: list[str | Path] | None = None,
        sampling_frequency: float,
    ):
        if folder_path is not None and file_path_list is not None:
            raise ValueError("Provide either 'folder_path' or 'file_path_list', not both.")
        if folder_path is None and file_path_list is None:
            raise ValueError("Provide one of 'folder_path' or 'file_path_list'.")

        if folder_path is not None:
            folder_path = Path(folder_path)
            cel_files = sorted(folder_path.glob("*.CEL"))
            if len(cel_files) == 0:
                raise ValueError(f"No .CEL files found in {folder_path}")
        else:
            cel_files = [Path(f) for f in file_path_list]

        unit_ids = []
        unit_names = []
        cluster_ids = []
        spike_times_dict = {}

        for cel_file in cel_files:
            cluster_id, spike_times_seconds = XClustSortingExtractor._parse_cel_file(cel_file)
            # XClust filenames follow the pattern {session_type}~{cluster_number}.CEL
            # e.g. BL1~1.CEL, ESA23D~2.CEL. We split on "~" to build clean identifiers.
            file_stem = cel_file.stem
            session_type, cluster_number = file_stem.split("~")
            # unit_id: unique identifier with "~" replaced by "_" to avoid path/query issues
            unit_id = f"{session_type}_{cluster_number}"
            # unit_name: human-readable label clarifying that the number refers to a cluster
            unit_name = f"{session_type}_cluster_{cluster_number}"
            unit_ids.append(unit_id)
            unit_names.append(unit_name)
            cluster_ids.append(cluster_id)
            spike_times_dict[unit_id] = np.unique(spike_times_seconds)

        BaseSorting.__init__(self, sampling_frequency=sampling_frequency, unit_ids=unit_ids)

        self.add_sorting_segment(XClustSortingSegment(spike_times_dict, sampling_frequency))
        self.set_property("unit_name", np.array(unit_names))
        # cluster_id is not unique across session types (e.g. BL1~1 and ESA23D~1 both have
        # cluster_id "1"), so it is stored as a property for provenance rather than as unit_id.
        self.set_property("cluster_id", np.array(cluster_ids))

        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()) if folder_path is not None else None,
            "file_path_list": [str(Path(f).absolute()) for f in file_path_list] if file_path_list is not None else None,
            "sampling_frequency": sampling_frequency,
        }

    @staticmethod
    def _parse_cel_file(file_path):
        """Parse an XClust .CEL file and return the cluster ID and spike times in seconds.

        Parameters
        ----------
        file_path : str or Path
            Path to a `.CEL` file.

        Returns
        -------
        cluster_id : int
            The cluster ID from the header.
        spike_times : numpy.ndarray
            1-D array of spike times in seconds.
        """
        file_path = Path(file_path)
        cluster_id = None
        fields = None

        with open(file_path, "r") as f:
            in_header = False
            for line in f:
                stripped = line.strip()
                if stripped == "%%BEGINHEADER":
                    in_header = True
                    continue
                if stripped == "%%ENDHEADER":
                    break
                if in_header:
                    if stripped.startswith("% Cluster:"):
                        cluster_id = stripped.split(":")[-1].strip()
                    elif stripped.startswith("% Fields:"):
                        fields = stripped.split(":")[-1].strip().split()

        if cluster_id is None:
            raise ValueError(f"No 'Cluster' field found in header of {file_path}")
        if fields is None:
            raise ValueError(f"No 'Fields' line found in header of {file_path}")
        if "time" not in fields:
            raise ValueError(f"No 'time' field found in Fields of {file_path}")

        time_column_index = fields.index("time")

        data = np.loadtxt(file_path, comments="%")
        spike_times = data[:, time_column_index] if data.ndim == 2 else np.array([data[time_column_index]])

        return cluster_id, spike_times


class XClustSortingSegment(BaseSortingSegment):
    def __init__(self, spike_times_dict, sampling_frequency):
        BaseSortingSegment.__init__(self)
        self._spike_times_dict = spike_times_dict
        self._sampling_frequency = sampling_frequency

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        start_time = None if start_frame is None else start_frame / self._sampling_frequency
        end_time = None if end_frame is None else end_frame / self._sampling_frequency

        spike_times = self.get_unit_spike_train_in_seconds(unit_id=unit_id, start_time=start_time, end_time=end_time)
        frames = np.round(spike_times * self._sampling_frequency).astype("int64", copy=False)
        return np.unique(frames)

    def get_unit_spike_train_in_seconds(self, unit_id, start_time=None, end_time=None):
        # XClust .CEL files store spike times natively in seconds
        spike_times = self._spike_times_dict[unit_id]

        if start_time is not None:
            start_index = np.searchsorted(spike_times, start_time, side="left")
        else:
            start_index = 0

        if end_time is not None:
            end_index = np.searchsorted(spike_times, end_time, side="left")
        else:
            end_index = spike_times.size

        return spike_times[start_index:end_index]


read_xclust = define_function_from_class(source_class=XClustSortingExtractor, name="read_xclust")
