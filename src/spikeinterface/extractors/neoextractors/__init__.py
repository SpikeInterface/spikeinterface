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

neo_recording_extractors_list = [
    AlphaOmegaRecordingExtractor,
    AxonaRecordingExtractor,
    BiocamRecordingExtractor,
    BlackrockRecordingExtractor,
    CedRecordingExtractor,
    EDFRecordingExtractor,
    IntanRecordingExtractor,
    MaxwellRecordingExtractor,
    MEArecRecordingExtractor,
    MCSRawRecordingExtractor,
    NeuralynxRecordingExtractor,
    NeuroScopeRecordingExtractor,
    NixRecordingExtractor,
    OpenEphysBinaryRecordingExtractor,
    OpenEphysLegacyRecordingExtractor,
    PlexonRecordingExtractor,
    Plexon2RecordingExtractor,
    Spike2RecordingExtractor,
    SpikeGadgetsRecordingExtractor,
    SpikeGLXRecordingExtractor,
    TdtRecordingExtractor,
    NeuroExplorerRecordingExtractor,
]

neo_sorting_extractors_list = [
    BlackrockSortingExtractor,
    MEArecSortingExtractor,
    NeuralynxSortingExtractor,
    Plexon2SortingExtractor,
]

neo_event_extractors_list = [
    AlphaOmegaEventExtractor,
    OpenEphysBinaryEventExtractor,
    Plexon2EventExtractor,
    SpikeGLXEventExtractor,
]
