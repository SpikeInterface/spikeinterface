from copy import deepcopy

from .amplitude_scalings import AmplitudeScalingsCalculator
from .spike_amplitudes import SpikeAmplitudesCalculator


def get_postprocessing_list():
    return deepcopy()


def resolve_postprocessing_graph(pipelines):
    pass


def compute_postprocessing(waveform_extractor, postprocessing_names, **kwargs):
    pass


_postprocessing_names_to_classes = dict(
    amplitude_scaling=AmplitudeScalingsCalculator, spike_amplitudes=SpikeAmplitudesCalculator
)
