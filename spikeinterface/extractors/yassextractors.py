import numpy as np
from pathlib import Path

from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)

try:
    import yaml

    HAVE_YAML = True
except:
    HAVE_YAML = False


class YassSortingExtractor(BaseSorting):
    extractor_name = 'YassExtractor'
    mode = 'folder'
    installed = HAVE_YAML  # check at class level if installed or not

    has_default_locations = False
    is_writable = False
    installation_mesg = "To use the Yass extractor, install pyyaml: \n\n pip install pyyaml\n\n"  # error message when not installed

    def __init__(self, folder_path):
        assert HAVE_YAML, self.installation_mesg

        folder_path = Path(folder_path)

        self.fname_spike_train = folder_path / 'tmp' / 'output' / 'spike_train.npy'
        self.fname_templates = folder_path / 'tmp' / 'output' / 'templates' / 'templates_0sec.npy'
        self.fname_config = folder_path / 'config.yaml'

        # Read CONFIG File
        with open(self.fname_config, 'r') as stream:
            self.config = yaml.safe_load(stream)

        spiketrains = np.load(self.fname_spike_train)
        unit_ids = np.unique(spiketrains[:, 1])

        # initialize
        sampling_frequency = self.config['recordings']['sampling_rate']
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(YassSortingSegment(spiketrains))

        self._kwargs = {'folder_path': str(folder_path)}


class YassSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains):
        BaseSortingSegment.__init__(self)
        # spiketrains is a 2 columns
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        mask = self._spiketrains == unit_id
        times = self._spiketrains[mask, 0].squeeze()
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


def read_yass(*args, **kwargs):
    sorting = YassSortingExtractor(*args, **kwargs)
    return sorting


read_yass.__doc__ = YassSortingExtractor.__doc__
