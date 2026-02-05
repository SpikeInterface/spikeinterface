from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class SpykingCircusSortingExtractor(BaseSorting):
    """Load SpykingCircus format data as a recording extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the SpykingCircus folder.

    Returns
    -------
    extractor : SpykingCircusSortingExtractor
        Loaded data.
    """

    installation_mesg = "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, folder_path):
        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        spykingcircus_folder = Path(folder_path)
        listfiles = spykingcircus_folder.iterdir()

        parent_folder = None
        result_folder = None
        for f in listfiles:
            if f.is_dir():
                if any([f_.suffix == ".hdf5" for f_ in f.iterdir()]):
                    parent_folder = spykingcircus_folder
                    result_folder = f

        if parent_folder is None:
            parent_folder = spykingcircus_folder.parent
            for f in parent_folder.iterdir():
                if f.is_dir():
                    if any([f_.suffix == ".hdf5" for f_ in f.iterdir()]):
                        result_folder = spykingcircus_folder

        assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), "Not a valid spyking circus folder"

        # load files
        results = None
        for f in result_folder.iterdir():
            if "result.hdf5" in str(f):
                results = f
            if "result-merged.hdf5" in str(f):
                results = f
                break
        if results is None:
            raise Exception(spykingcircus_folder, " is not a spyking circus folder")

        # load params
        sample_rate = None
        for f in parent_folder.iterdir():
            if f.suffix == ".params":
                sample_rate = _load_sample_rate(f)

        assert sample_rate is not None, "sample rate not found"

        with h5py.File(results, "r") as f_results:
            spiketrains = []
            unit_ids = []
            for temp in f_results["spiketimes"].keys():
                spiketrains.append(np.array(f_results["spiketimes"][temp]).astype("int64"))
                unit_ids.append(int(temp.split("_")[-1]))

        BaseSorting.__init__(self, sample_rate, unit_ids)
        self.add_sorting_segment(SpykingcircustSortingSegment(unit_ids, spiketrains))

        self._kwargs = {"folder_path": str(Path(folder_path).absolute())}
        self.extra_requirements.append("h5py")


class SpykingcircustSortingSegment(BaseSortingSegment):
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


def _load_sample_rate(params_file):
    sample_rate = None
    with params_file.open("r") as f:
        for r in f.readlines():
            if "sampling_rate" in r:
                sample_rate = r.split("=")[-1]
                if "#" in sample_rate:
                    sample_rate = sample_rate[: sample_rate.find("#")]
                sample_rate = float(sample_rate)
    return sample_rate


read_spykingcircus = define_function_from_class(source_class=SpykingCircusSortingExtractor, name="read_spykingcircus")
