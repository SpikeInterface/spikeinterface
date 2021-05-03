import unittest


import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.extractors import *

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite

class MearecRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecRecordingExtractor
    downloads = ['mearec']
    entities = ['mearec/mearec_test_10s.h5']

class MearecSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecSortingExtractor
    downloads = ['mearec']
    entities = ['mearec/mearec_test_10s.h5']


class SpikeGLXRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = SpikeGLXRecordingExtractor
    downloads = ['spikeglx']
    entities = [
        ('spikeglx/Noise4Sam_g0', {'stream_id': 'imec0.ap'}),
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
    ]

class OpenEphysLegacyRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysLegacyRecordingExtractor
    downloads = ['openephys']
    entities = [
        'openephys/OpenEphys_SampleData_1',
    ]

class ItanRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
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

class PlexonRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = PlexonRecordingExtractor
    downloads = ['plexon']
    entities = [
        'plexon/File_plexon_3.plx',
    ]

# TODO : this fail, need investigate
# class NeuralynxRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    # ExtractorClass = NeuralynxRecordingExtractor
    # downloads = ['neuralynx']
    # entities = [
        # 'neuralynx/Cheetah_v5.6.3',
    # ]


class BlackrockRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BlackrockRecordingExtractor
    downloads = ['blackrock']
    entities = [
        'blackrock/FileSpec2.3001.ns5',
        ('blackrock/blackrock_2_1/l101210-001.ns2', {'stream_id': '2'}),
        ('blackrock/blackrock_2_1/l101210-001.ns2', {'stream_id': '5'}),
    ]

class MCSRawRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MCSRawRecordingExtractor
    downloads = ['rawmcs']
    entities = [
        'rawmcs/raw_mcs_with_header_1.raw',
    ]


class KiloSortSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = KiloSortSortingExtractor
    downloads = ['phy']
    entities = [
        'phy/phy_example_0',
    ]

@pytest.mark.skip(reason="Maxwell HDF5 compression need a manual installable plugin!!!")
class MaxwellRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MaxwellRecordingExtractor
    downloads = ['maxwell']
    entities = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        ('maxwell/MaxTwo_data/Network/000028/data.raw.h5', {'stream_id':'well0000', 'rec_name': 'rec0000'})
    ]


if __name__ == '__main__':
    #~ test = MearecRecordingTest()
    #~ test = MearecSortingTest()
    #~ test = SpikeGLXRecordingTest()
    #~ test = OpenEphysBinaryRecordingTest()
    # test = OpenEphysLegacyRecordingTest()
    # test = ItanRecordingTest()
    # test = NeuroScopeRecordingTest()
    # test = PlexonRecordingTest()
    # test = NeuralynxRecordingTest()
    # test = BlackrockRecordingTest()
    # test = MCSRawRecordingTest()
    # test = KiloSortSortingTest()
    test = MaxwellRecordingTest()
    
    test.setUp()
    test.test_open()
    
    

