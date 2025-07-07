from .alphaomega import AlphaOmegaRecordingExtractor, AlphaOmegaEventExtractor, read_alphaomega, read_alphaomega_event
from .axon import AxonRecordingExtractor, read_axon
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
    AlphaOmegaRecordingExtractor: dict(wrapper_string="read_alphaomega", wrapper_class=read_alphaomega),
    AxonRecordingExtractor: dict(wrapper_string="read_axon", wrapper_class=read_axon),
    AxonaRecordingExtractor: dict(wrapper_string="read_axona", wrapper_class=read_axona),
    BiocamRecordingExtractor: dict(wrapper_string="read_biocam", wrapper_class=read_biocam),
    BlackrockRecordingExtractor: dict(wrapper_string="read_blackrock", wrapper_class=read_blackrock),
    CedRecordingExtractor: dict(wrapper_string="read_ced", wrapper_class=read_ced),
    EDFRecordingExtractor: dict(wrapper_string="read_edf", wrapper_class=read_edf),
    IntanRecordingExtractor: dict(wrapper_string="read_intan", wrapper_class=read_intan),
    MaxwellRecordingExtractor: dict(wrapper_string="read_maxwell", wrapper_class=read_maxwell),
    MEArecRecordingExtractor: dict(wrapper_string="read_mearec", wrapper_class=read_mearec),
    MCSRawRecordingExtractor: dict(wrapper_string="read_mcsraw", wrapper_class=read_mcsraw),
    NeuralynxRecordingExtractor: dict(wrapper_string="read_neuralynx", wrapper_class=read_neuralynx),
    NeuroScopeRecordingExtractor: dict(
        wrapper_string="read_neuroscope_recording", wrapper_class=read_neuroscope_recording
    ),
    NeuroNexusRecordingExtractor: dict(wrapper_string="read_neuronexus", wrapper_class=read_neuronexus),
    NixRecordingExtractor: dict(wrapper_string="read_nix", wrapper_class=read_nix),
    OpenEphysBinaryRecordingExtractor: dict(wrapper_string="read_openephys", wrapper_class=read_openephys),
    OpenEphysLegacyRecordingExtractor: dict(wrapper_string="read_openephys", wrapper_class=read_openephys),
    PlexonRecordingExtractor: dict(wrapper_string="read_plexon", wrapper_class=read_plexon),
    Plexon2RecordingExtractor: dict(wrapper_string="read_plexon2", wrapper_class=read_plexon2),
    Spike2RecordingExtractor: dict(wrapper_string="read_spike2", wrapper_class=read_spike2),
    SpikeGadgetsRecordingExtractor: dict(wrapper_string="read_spikegadgets", wrapper_class=read_spikegadgets),
    SpikeGLXRecordingExtractor: dict(wrapper_string="read_spikeglx", wrapper_class=read_spikeglx),
    TdtRecordingExtractor: dict(wrapper_string="read_tdt", wrapper_class=read_tdt),
    NeuroExplorerRecordingExtractor: dict(wrapper_string="read_neuroexplorer", wrapper_class=read_neuroexplorer),
}

neo_sorting_extractors_dict = {
    BlackrockSortingExtractor: dict(wrapper_string="read_blackrock_sorting", wrapper_class=read_blackrock_sorting),
    MEArecSortingExtractor: dict(wrapper_string="read_mearec", wrapper_class=read_mearec),
    NeuralynxSortingExtractor: dict(wrapper_string="read_neuralynx_sorting", wrapper_class=read_neuralynx_sorting),
    PlexonSortingExtractor: dict(wrapper_string="read_plexon_sorting", wrapper_class=read_plexon_sorting),
    Plexon2SortingExtractor: dict(wrapper_string="read_plexon2_sorting", wrapper_class=read_plexon2_sorting),
    NeuroScopeSortingExtractor: dict(wrapper_string="read_neuroscope_sorting", wrapper_class=read_neuroscope_sorting),
}

neo_event_extractors_dict = {
    AlphaOmegaEventExtractor: dict(wrapper_string="read_alphaomega_event", wrapper_class=read_alphaomega_event),
    OpenEphysBinaryEventExtractor: dict(wrapper_string="read_openephys_event", wrapper_class=read_openephys_event),
    Plexon2EventExtractor: dict(wrapper_string="read_plexon2_event", wrapper_class=read_plexon2_event),
    SpikeGLXEventExtractor: dict(wrapper_string="read_spikeglx_event", wrapper_class=read_spikeglx_event),
    MaxwellEventExtractor: dict(wrapper_string="read_maxwell_event", wrapper_class=read_maxwell_event),
}


# Utils dicts used for get_neo_extractor and get_neo_streams
neo_recording_class_dict = {
    rec_class.__name__.replace("Recording", "").replace("Extractor", "").lower(): rec_class
    for rec_class in neo_recording_extractors_dict.keys()
}
neo_sorting_class_dict = {
    sort_class.__name__.replace("Sorting", "").replace("Extractor", "").lower(): sort_class
    for sort_class in neo_sorting_extractors_dict.keys()
}
neo_event_class_dict = {
    event_class.__name__.replace("Event", "").replace("Extractor", "").lower(): event_class
    for event_class in neo_event_extractors_dict.keys()
}
