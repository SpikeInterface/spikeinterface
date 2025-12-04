from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class HerdingspikesSortingExtractor(BaseSorting):
    """Load HerdingSpikes format data as a sorting extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the ALF folder.
    load_unit_info : bool, default: True
        Whether to load the unit info from the file.

    Returns
    -------
    extractor : HerdingSpikesSortingExtractor
        The loaded data.
    """

    installation_mesg = "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, file_path):
        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        self._recording_file = file_path
        self._rf = h5py.File(self._recording_file, mode="r")
        if "Sampling" in self._rf:
            if self._rf["Sampling"][()] == 0:
                sampling_frequency = None
            else:
                sampling_frequency = self._rf["Sampling"][()]

        spike_ids = self._rf["cluster_id"][()]
        unit_ids = np.unique(spike_ids)
        spike_times = self._rf["times"][()]
        unit_locs = self._rf["centres"][()]

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(HerdingspikesSortingSegment(unit_ids, spike_times, spike_ids))

        self.set_property("hs_location", unit_locs)

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

        self.extra_requirements.append("h5py")


# alias for backward compatiblity
HS2SortingExtractor = HerdingspikesSortingExtractor


class HerdingspikesSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spike_times, spike_ids):
        BaseSortingSegment.__init__(self)
        # spike_times is a dict
        self._unit_ids = list(unit_ids)
        self._spike_times = spike_times
        self._spike_ids = spike_ids

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        mask = self._spike_ids == unit_id
        times = self._spike_times[mask]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_herdingspikes = define_function_from_class(source_class=HerdingspikesSortingExtractor, name="read_herdingspikes")
