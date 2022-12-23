import unittest

import pytest
import numpy as np

from spikeinterface import get_global_dataset_folder
from spikeinterface.extractors import *

from spikeinterface.extractors.tests.common_tests import (RecordingCommonTestSuite,
                                                          SortingCommonTestSuite, EventCommonTestSuite)

local_folder = get_global_dataset_folder() / 'ephy_testing_data'


class MearecRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecRecordingExtractor
    downloads = ['mearec']
    entities = ['mearec/mearec_test_10s.h5']
    neo_funcs = dict()


class MearecSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecSortingExtractor
    downloads = ['mearec']
    entities = ['mearec/mearec_test_10s.h5']


class SpikeGLXRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = SpikeGLXRecordingExtractor
    downloads = ['spikeglx']
    entities = [
        ('spikeglx/Noise4Sam_g0', {'stream_id': 'imec0.ap'}),
        ('spikeglx/Noise4Sam_g0', {'stream_id': 'imec0.ap', 'load_sync_channel': True}),
        ('spikeglx/Noise4Sam_g0', {'stream_id': 'imec0.lf'}),
        ('spikeglx/Noise4Sam_g0', {'stream_id': 'nidq'}),
    ]


class OpenEphysBinaryRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysBinaryRecordingExtractor
    downloads = ['openephysbinary']
    entities = [
        'openephysbinary/v0.4.4.1_with_video_tracking',
        ('openephysbinary/v0.5.3_two_neuropixels_stream', {'stream_id': '0'}),
        ('openephysbinary/v0.5.3_two_neuropixels_stream', {'stream_id': '1'}),
        ('openephysbinary/v0.5.x_two_nodes', {'stream_id': '0'}),
        ('openephysbinary/v0.5.x_two_nodes', {'stream_id': '1'}),
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'stream_id': '0', 'block_index': 0}),
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'stream_id': '1', 'block_index': 1}),
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'stream_id': '2', 'block_index': 2}),
    ]


class OpenEphysBinaryEventTest(EventCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysBinaryEventExtractor
    downloads = ['openephysbinary']
    entities = [
        'openephysbinary/v0.4.4.1_with_video_tracking',
        'openephysbinary/v0.5.3_two_neuropixels_stream',
        'openephysbinary/v0.5.x_two_nodes',
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'block_index': 0}),
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'block_index': 1}),
        ('openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
        {'block_index': 2}),
    ]


class OpenEphysLegacyRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysLegacyRecordingExtractor
    downloads = ['openephys']
    entities = [
        'openephys/OpenEphys_SampleData_1',
    ]


class IntanRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = IntanRecordingExtractor
    downloads = ['intan']
    entities = [
        ('intan/intan_rhd_test_1.rhd', {'stream_id': '0'}),
        ('intan/intan_rhd_test_1.rhd', {'stream_id': '2'}),
        ('intan/intan_rhd_test_1.rhd', {'stream_id': '3'}),
        ('intan/intan_rhs_test_1.rhs', {'stream_id': '0'}),
        ('intan/intan_rhs_test_1.rhs', {'stream_id': '3'}),
        ('intan/intan_rhs_test_1.rhs', {'stream_id': '4'}),
        ('intan/intan_rhs_test_1.rhs', {'stream_id': '11'}),
    ]


class NeuroScopeRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuroScopeRecordingExtractor
    downloads = ['neuroscope']
    entities = [
        'neuroscope/test1/test1.xml',
    ]


class NeuroScopeSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuroScopeSortingExtractor
    downloads = ['neuroscope']
    entities = [
        'neuroscope/dataset_1',
        {'resfile_path': local_folder / 'neuroscope/dataset_1/YutaMouse42-15111710.res.1',
         'clufile_path': local_folder / 'neuroscope/dataset_1/YutaMouse42-15111710.clu.1'},
    ]


class PlexonRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = PlexonRecordingExtractor
    downloads = ['plexon']
    entities = [
        'plexon/File_plexon_3.plx',
    ]


class NeuralynxRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuralynxRecordingExtractor
    downloads = ['neuralynx']
    entities = [
        'neuralynx/Cheetah_v1.1.0/original_data',
        'neuralynx/Cheetah_v4.0.2/original_data',
        'neuralynx/Cheetah_v5.4.0/original_data',
        'neuralynx/Cheetah_v5.5.1/original_data',
        'neuralynx/Cheetah_v5.6.3/original_data',
        'neuralynx/Cheetah_v5.7.4/original_data',
    ]

class NeuralynxSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuralynxSortingExtractor
    downloads = ['neuralynx']
    entities = [
        'neuralynx/Cheetah_v5.5.1/original_data',
        'neuralynx/Cheetah_v5.6.3/original_data',
    ]


class BlackrockRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BlackrockRecordingExtractor
    downloads = ['blackrock']
    entities = [
        'blackrock/FileSpec2.3001.ns5',
        ('blackrock/blackrock_2_1/l101210-001.ns2', {'stream_id': '2'}),
        ('blackrock/blackrock_2_1/l101210-001.ns2', {'stream_id': '5'}),
    ]


class BlackrockSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BlackrockSortingExtractor
    downloads = ['blackrock']
    entities = [
        'blackrock/FileSpec2.3001.nev',
        "blackrock/blackrock_2_1/l101210-001.nev"
    ]


class MCSRawRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MCSRawRecordingExtractor
    downloads = ['rawmcs']
    entities = [
        'rawmcs/raw_mcs_with_header_1.raw',
    ]


class TdTRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = TdtRecordingExtractor
    downloads = ['tdt']
    entities = [
        ('tdt/aep_05', {'stream_id': '1'})
    ]


class AxonaRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = AxonaRecordingExtractor
    downloads = ['axona']
    entities = [
        'axona/axona_raw',
    ]


class KiloSortSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = KiloSortSortingExtractor
    downloads = ['phy']
    entities = [
        'phy/phy_example_0',
    ]


class Spike2RecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = Spike2RecordingExtractor
    downloads = ['spike2/130322-1LY.smr']
    entities = [
        ('spike2/130322-1LY.smr', {'stream_id': '1'}),
    ]


class CedRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CedRecordingExtractor
    downloads = [
        'spike2/130322-1LY.smr',
        'spike2/m365_1sec.smrx',
    ]
    entities = [
        ('spike2/130322-1LY.smr', {'stream_id': '1'}),
        'spike2/m365_1sec.smrx',
    ]


class MaxwellRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MaxwellRecordingExtractor
    downloads = ['maxwell']
    entities = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        ('maxwell/MaxTwo_data/Network/000028/data.raw.h5',
         {'stream_id': 'well000', 'rec_name': 'rec0000'})
    ]

    def setUp(self):
        from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin
        auto_install_maxwell_hdf5_compression_plugin()
        return super().setUp()


class SpikeGadgetsRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = SpikeGadgetsRecordingExtractor
    downloads = ['spikegadgets']
    entities = [
        ('spikegadgets/20210225_em8_minirec2_ac.rec', {'stream_id': 'ECU'}),
        ('spikegadgets/20210225_em8_minirec2_ac.rec', {'stream_id': 'trodes'}),
        'spikegadgets/W122_06_09_2019_1_fromSD.rec'
    ]


class BiocamRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BiocamRecordingExtractor
    downloads = ['biocam/biocam_hw3.0_fw1.6.brw']
    entities = [
        'biocam/biocam_hw3.0_fw1.6.brw'
    ]


class AlphaOmegaRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = AlphaOmegaRecordingExtractor
    downloads = ["alphaomega"]
    entities = [
        "alphaomega/mpx_map_version4",
    ]


class AlphaOmegaEventTest(EventCommonTestSuite, unittest.TestCase):
    ExtractorClass = AlphaOmegaEventExtractor
    downloads = ["alphaomega"]
    entities = [
        "alphaomega/mpx_map_version4",
    ]


class EDFRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = EDFRecordingExtractor
    downloads = ['edf']
    entities = ['edf/edf+C.edf']


class CellExplorerSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CellExplorerSortingExtractor
    downloads = ['cellexplorer']
    entities = [
        'cellexplorer/dataset_1/20170311_684um_2088um_170311_134350.spikes.cellinfo.mat',
        ('cellexplorer/dataset_2/20170504_396um_0um_merge.spikes.cellinfo.mat', 
        {'session_info_matfile_path': 
            local_folder / 'cellexplorer/dataset_2/20170504_396um_0um_merge.sessionInfo.mat'})
    ]


if __name__ == '__main__':
    # test = MearecSortingTest()
    # test = SpikeGLXRecordingTest()
    # test = OpenEphysBinaryRecordingTest()
    # test = SpikeGLXRecordingTest()
    # test = OpenEphysBinaryRecordingTest()
    # test = OpenEphysLegacyRecordingTest()
    test = CellExplorerSortingTest()
    # test = ItanRecordingTest()
    # test = NeuroScopeRecordingTest()
    # test = PlexonRecordingTest()
    # test = NeuralynxRecordingTest()
    # test = BlackrockRecordingTest()
    # test = MCSRawRecordingTest()
    # test = KiloSortSortingTest()
    # test = Spike2RecordingTest()
    # test = CedRecordingTest()
    # test = MaxwellRecordingTest()
    # test = SpikeGadgetsRecordingTest()
    # test = NeuroScopeSortingTest()

    test.setUp()
    test.test_open()
