from __future__ import annotations

from pathlib import Path
import re
import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class MClustSortingExtractor(BaseSorting):
    """Load MClust sorting solution as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder with t files.
    sampling_frequency : sampling frequency
        sampling frequency in Hz.
    sampling_frequency_raw: float or None, default: None
        Required to read files with raw formats. In that case, the samples are saved in the same
        unit as the input data
        Examples:
            - If raw time is in tens of ms sampling_frequency_raw=10000
            - If raw time is in samples sampling_frequency_raw=sampling_frequency
    Returns
    -------
    extractor : MClustSortingExtractor
        Loaded data.
    """

    def __init__(self, folder_path, sampling_frequency, sampling_frequency_raw=None):
        end_header_str = "%%ENDHEADER"
        ext_list = ["t64", "t32", "t", "raw64", "raw32"]
        unit_ids = []
        ext = None

        for e in ext_list:
            files = Path(folder_path).glob(f"*.{e}")
            if files:
                ext = e
                break

        if ext is None:
            raise Exception("Mclust files not found in path")

        if ext.startswith("raw") and sampling_frequency_raw is None:
            raise Exception(f"To load files with extension {ext} a sampling_frequency_raw input is required.")

        if ext.endswith("64"):
            dataformat = ">u8"
        else:
            dataformat = ">u4"

        spiketrains = {}

        for filename in files:
            unit = int(re.search("_([0-9]+?)$", filename.stem).group(1))
            unit_ids.append(unit)
            with open(filename, "rb") as f:
                reading_header = True
                while reading_header:
                    line = f.readline()
                    reading_header = not line.decode("utf-8").startswith(end_header_str)
                times = np.fromfile(f, dtype=dataformat)
            if ext.startswith("t"):
                times = times / 10000
            else:
                times = times / sampling_frequency_raw
            spiketrains[unit] = np.rint(times * sampling_frequency)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.add_sorting_segment(MClustSortingSegment(unit_ids, spiketrains))
        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()),
            "sampling_frequency": sampling_frequency,
            "sampling_frequency_raw": sampling_frequency_raw,
        }


class MClustSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spiketrains):
        BaseSortingSegment.__init__(self)
        self._unit_ids = list(unit_ids)
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._spiketrains[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_mclust = define_function_from_class(source_class=MClustSortingExtractor, name="read_mclust")
