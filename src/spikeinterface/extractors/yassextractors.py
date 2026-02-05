from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class YassSortingExtractor(BaseSorting):
    """Load YASS format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the ALF folder.

    Returns
    -------
    extractor : YassSortingExtractor
        Loaded data.
    """

    installation_mesg = "To use the Yass extractor, install pyyaml: \n\n pip install pyyaml\n\n"

    def __init__(self, folder_path):
        try:
            import yaml
        except:
            raise ImportError(self.installation_mesg)

        folder_path = Path(folder_path)

        self.fname_spike_train = folder_path / "tmp" / "output" / "spike_train.npy"
        self.fname_templates = folder_path / "tmp" / "output" / "templates" / "templates_0sec.npy"
        self.fname_config = folder_path / "config.yaml"

        # Read CONFIG File
        with open(self.fname_config, "r") as stream:
            self.config = yaml.safe_load(stream)

        spiketrains = np.load(self.fname_spike_train)
        unit_ids = np.unique(spiketrains[:, 1])

        # initialize
        sampling_frequency = self.config["recordings"]["sampling_rate"]
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(YassSortingSegment(spiketrains))

        self._kwargs = {"folder_path": str(Path(folder_path).absolute())}
        self.extra_requirements.append("pyyaml")


class YassSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains):
        BaseSortingSegment.__init__(self)
        # spiketrains is a 2 columns
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        mask = self._spiketrains[:, 1] == unit_id
        times = self._spiketrains[mask, 0].squeeze()
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_yass = define_function_from_class(source_class=YassSortingExtractor, name="read_yass")
