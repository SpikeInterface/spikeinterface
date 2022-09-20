from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, NumpySorting
from spikeinterface.comparison import AddTemplatesRecording
from spikeinterface.extractors.toy_example import synthesize_random_firings


class HybridUnitsRecording(AddTemplatesRecording):

    def __init__(self, templates: np.ndarray, target_recording: Union[BaseRecording, None] = None,
                 n_before: Union[List[int], int, None] = None, frequency: float = 10,
                 amplitude_std: float = 0.0, refrac_period: float = 2.0):
        """
        TODO
        """

        t_max = target_recording.get_num_frames() # TODO: What if target_recording is None
        fs = target_recording.sampling_frequency # TODO: What if target_recording is None
        n_units = len(templates)

        # Making the sorting object.
        spike_times, spike_labels = synthesize_random_firings(num_units=n_units, sampling_frequency=fs, duration=t_max) # TODO: refrac_period missing
        spike_trains = {unit_id: spike_times[spike_labels == unit_id] for unit_id in range(n_units)}
        sorting = NumpySorting.from_dict(spike_trains)

        amplitude_factor = [np.random.normal(loc=1.0, scale=amplitude_std, size=len(sorting.get_unit_spike_train(unit_id))) for unit_id in sorting.unit_ids]

        AddTemplatesRecording.__init__(self, sorting, templates, nbefore, amplitude_factor, target_recording, t_max)
