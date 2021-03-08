import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import (
        SpikeGLXRecordingExtractor,
        MEArecRecordingExtractor,MEArecSortingExtractor,
        OpenEphysLegacyRecordingExtractor, OpenEphysBinaryRecordingExtractor,
        IntanRecordingExtractor,
        )
        

from spikeinterface.extractors.tests.file_retrieve import download_test_file


# TODO use datalad and make this folder at API
basefolder = '/home/samuel/Documents/ephy_testing_data/'



def test_mearec_extractors():
    mearec_file = basefolder + 'mearec/mearec_test_10s.h5'
    
    rec = MEArecRecordingExtractor(mearec_file)
    print(rec)
    
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


if __name__ == '__main__':
    #~ test_mearec_extractors()
    #~ test_spikeglx_extractors()
    #~ test_openephy_legacy()
    #~ test_openephy_binary()
    test_intan()
