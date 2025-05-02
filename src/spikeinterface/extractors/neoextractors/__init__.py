from .alphaomega import AlphaOmegaRecordingExtractor, AlphaOmegaEventExtractor, read_alphaomega, read_alphaomega_event
from .axona import AxonaRecordingExtractor, read_axona
from .biocam import BiocamRecordingExtractor, read_biocam
from .blackrock import BlackrockRecordingExtractor, BlackrockSortingExtractor, read_blackrock, read_blackrock_sorting
from .ced import CedRecordingExtractor, read_ced
from .edf import EDFRecordingExtractor, read_edf
from .intan import IntanRecordingExtractor, read_intan
from .maxwell import MaxwellRecordingExtractor, MaxwellEventExtractor, read_maxwell, read_maxwell_event
from .mearec import MEArecRecordingExtractor, MEArecSortingExtractor, read_mearec
from .mcsraw import MCSRawRecordingExtractor, read_mcsraw
from .neuralynx import NeuralynxRecordingExtractor, NeuralynxSortingExtractor, read_neuralynx, read_neuralynx_sorting
from .neuronexus import NeuroNexusRecordingExtractor, read_neuronexus
from .neuroscope import (
    NeuroScopeRecordingExtractor,
    NeuroScopeSortingExtractor,
    read_neuroscope_recording,
    read_neuroscope_sorting,
    read_neuroscope,
)
from .neuroexplorer import NeuroExplorerRecordingExtractor, read_neuroexplorer
from .nix import NixRecordingExtractor, read_nix
from .openephys import (
    OpenEphysLegacyRecordingExtractor,
    OpenEphysBinaryRecordingExtractor,
    OpenEphysBinaryEventExtractor,
    read_openephys,
    read_openephys_event,
)
from .plexon import PlexonRecordingExtractor, PlexonSortingExtractor, read_plexon, read_plexon_sorting
from .plexon2 import (
    Plexon2SortingExtractor,
    Plexon2RecordingExtractor,
    Plexon2EventExtractor,
    read_plexon2,
    read_plexon2_sorting,
    read_plexon2_event,
)
from .spike2 import Spike2RecordingExtractor, read_spike2
from .spikegadgets import SpikeGadgetsRecordingExtractor, read_spikegadgets
from .spikeglx import SpikeGLXRecordingExtractor, SpikeGLXEventExtractor, read_spikeglx, read_spikeglx_event
from .tdt import TdtRecordingExtractor, read_tdt

from .neo_utils import get_neo_streams, get_neo_num_blocks

neo_recording_extractors_dict = {
    AlphaOmegaRecordingExtractor: read_alphaomega,
    AxonaRecordingExtractor: read_axona,
    BiocamRecordingExtractor: read_biocam,
    BlackrockRecordingExtractor: read_blackrock,
    CedRecordingExtractor: read_ced,
    EDFRecordingExtractor: read_edf,
    IntanRecordingExtractor: read_intan,
    MaxwellRecordingExtractor: read_maxwell,
    MEArecRecordingExtractor: read_mearec,
    MCSRawRecordingExtractor: read_mcsraw,
    NeuralynxRecordingExtractor: read_neuralynx,
    NeuroScopeRecordingExtractor: read_neuroscope_recording,
    NeuroNexusRecordingExtractor: read_neuronexus,
    NixRecordingExtractor: read_nix,
    OpenEphysBinaryRecordingExtractor: read_openephys,
    OpenEphysLegacyRecordingExtractor: read_openephys,
    PlexonRecordingExtractor: read_plexon,
    Plexon2RecordingExtractor: read_plexon2,
    Spike2RecordingExtractor: read_spike2,
    SpikeGadgetsRecordingExtractor: read_spikegadgets,
    SpikeGLXRecordingExtractor: read_spikeglx,
    TdtRecordingExtractor: read_tdt,
    NeuroExplorerRecordingExtractor: read_neuroexplorer,
}

neo_sorting_extractors_dict = {
    BlackrockSortingExtractor: read_blackrock_sorting,
    MEArecSortingExtractor: read_mearec,
    NeuralynxSortingExtractor: read_neuralynx_sorting,
    PlexonSortingExtractor: read_plexon_sorting,
    Plexon2SortingExtractor: read_plexon2_sorting,
    NeuroScopeSortingExtractor: read_neuroscope_sorting,
}

neo_event_extractors_dict = {
    AlphaOmegaEventExtractor: read_alphaomega_event,
    OpenEphysBinaryEventExtractor: read_openephys_event,
    Plexon2EventExtractor: read_plexon2_event,
    SpikeGLXEventExtractor: read_spikeglx_event,
    MaxwellEventExtractor: read_maxwell_event,
}

__all__ = [
    "neo_recording_extractors_dict",
    "neo_sorting_extractors_dict",
    "neo_event_extractors_dict",
    get_neo_streams.__name__,
    get_neo_num_blocks.__name__,
]
__all__ += [func.__name__ for func in neo_recording_extractors_dict.values()]
__all__ += [func.__name__ for func in neo_sorting_extractors_dict.values()]
__all__ += [func.__name__ for func in neo_event_extractors_dict.values()]
__all__.append("read_neuroscope")
