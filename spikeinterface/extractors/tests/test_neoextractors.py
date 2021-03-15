import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *





# TODO use datalad and make this folder at API
# from spikeinterface.extractors.tests.file_retrieve import download_test_file
basefolder = '/home/samuel/Documents/ephy_testing_data/'



def test_mearec_extractors():
    mearec_file = basefolder + 'mearec/mearec_test_10s.h5'
    
    rec = MEArecRecordingExtractor(mearec_file)
    print(rec)
    probe = rec.get_probe()
    print(probe)
    
    sorting = MEArecSortingExtractor(mearec_file, use_natural_unit_ids=True)
    print(sorting)
    print(sorting.get_unit_ids())

    sorting = MEArecSortingExtractor(mearec_file, use_natural_unit_ids=False)
    print(sorting)
    print(sorting.get_unit_ids())


def test_spikeglx_extractors():
    spikeglx_folder = basefolder + 'spikeglx/Noise4Sam_g0'
    rec = SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.ap')
    print(rec)

    probe = rec.get_probe()
    print(probe)
    

from spikeinterface.extractors.tests.file_retrieve import download_test_file

def test_openephys():
    oe_folder = basefolder +  'openephys/OpenEphys_SampleData_1'
    rec = OpenEphysLegacyRecordingExtractor(oe_folder)
    print(rec)
    
    oe_folder = basefolder +  'openephysbinary/v0.4.4.1_with_video_tracking/'
    rec = OpenEphysBinaryRecordingExtractor(oe_folder)
    print(rec)
    
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ traces = rec.get_traces()
    #~ ax.plot(traces[:32000*4, 5])
    #~ plt.show()

def test_intan():
    intan_file = basefolder +  'intan/intan_rhd_test_1.rhd'
    for stream_id in ('0', '2', '3'):
        rec = IntanRecordingExtractor(intan_file, stream_id=stream_id)
        print(rec)
    
    intan_file = basefolder +  'intan/intan_rhs_test_1.rhs'
    for stream_id in ('0', '11', '3', '4'):
        rec = IntanRecordingExtractor(intan_file, stream_id=stream_id)


def test_neuroscope():
    neuroscope_file = basefolder +  'neuroscope/test1/test1.xml'
    rec = NeuroScopeRecordingExtractor(neuroscope_file)
    print(rec)
    
def test_plexon():
    plexon_file = basefolder + 'plexon/File_plexon_3.plx'
    PlexonRecordingExtractor
    rec = PlexonRecordingExtractor(plexon_file)
    print(rec)

def test_neuralynx():
    neuralynx_folder = basefolder + 'neuralynx/Cheetah_v5.6.3/'
    rec = NeuralynxRecordingExtractor(neuralynx_folder)
    print(rec)

def test_blackrock():
    blackrock_file = basefolder + 'blackrock/FileSpec2.3001.ns5'
    rec = BlackrockRecordingExtractor(blackrock_file)
    print(rec)

def test_mcsraw():
    mcsraw_file = basefolder + 'rawmcs/raw_mcs_with_header_1.raw'
    rec = MCSRawRecordingExtractor(mcsraw_file)
    print(rec)

def test_kilosort():
    kilosort_folder = basefolder + 'phy/phy_example_0'
    sorting = KiloSortSortingExtractor(kilosort_folder)
    print(sorting)



if __name__ == '__main__':
    #~ test_mearec_extractors()
    test_spikeglx_extractors()
    #~ test_openephy_legacy()
    #~ test_openephy_binary()
    #~ test_intan()
    #~ test_neuroscope()
    #~ test_plexon()
    #~ test_neuralynx()
    #~ test_blackrock()
    #~ test_mcsraw()
    #~ test_kilosort()
    
    

