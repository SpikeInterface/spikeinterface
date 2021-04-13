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





#~ @pytest.mark.skip('')
#~ def test_intan():
    #~ intan_file = basefolder +  'intan/intan_rhd_test_1.rhd'
    #~ for stream_id in ('0', '2', '3'):
        #~ rec = IntanRecordingExtractor(intan_file, stream_id=stream_id)
        #~ print(rec)
    
    #~ intan_file = basefolder +  'intan/intan_rhs_test_1.rhs'
    #~ for stream_id in ('0', '11', '3', '4'):
        #~ rec = IntanRecordingExtractor(intan_file, stream_id=stream_id)

#~ @pytest.mark.skip('')
#~ def test_neuroscope():
    #~ neuroscope_file = basefolder +  'neuroscope/test1/test1.xml'
    #~ rec = NeuroScopeRecordingExtractor(neuroscope_file)
    #~ print(rec)

#~ @pytest.mark.skip('')
#~ def test_plexon():
    #~ plexon_file = basefolder + 'plexon/File_plexon_3.plx'
    #~ PlexonRecordingExtractor
    #~ rec = PlexonRecordingExtractor(plexon_file)
    #~ print(rec)

#~ @pytest.mark.skip('')
#~ def test_neuralynx():
    #~ neuralynx_folder = basefolder + 'neuralynx/Cheetah_v5.6.3/'
    #~ rec = NeuralynxRecordingExtractor(neuralynx_folder)
    #~ print(rec)

#~ @pytest.mark.skip('')
#~ def test_blackrock():
    #~ blackrock_file = basefolder + 'blackrock/FileSpec2.3001.ns5'
    #~ rec = BlackrockRecordingExtractor(blackrock_file)
    #~ print(rec)

#~ @pytest.mark.skip('')
#~ def test_mcsraw():
    #~ mcsraw_file = basefolder + 'rawmcs/raw_mcs_with_header_1.raw'
    #~ rec = MCSRawRecordingExtractor(mcsraw_file)
    #~ print(rec)

#~ @pytest.mark.skip('')
#~ def test_kilosort():
    #~ kilosort_folder = basefolder + 'phy/phy_example_0'
    #~ sorting = KiloSortSortingExtractor(kilosort_folder)
    #~ print(sorting)



if __name__ == '__main__':
    #~ test = MearecRecordingTest()
    #~ test = SpikeGLXRecordingTest()
    #~ test = OpenEphysBinaryRecordingTest()
    #~ test = OpenEphysLegacyRecordingTest()
    
    test.setUp()
    test.test_open()
    
    

