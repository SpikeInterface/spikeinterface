import unittest
import unittest
from pathlib import Path
from tempfile import mkdtemp
from datetime import datetime

import pytest
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries

from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class NwbRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbRecordingExtractor
    downloads = []
    entities = []


class NwbSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbSortingExtractor
    downloads = []
    entities = []
    

    
class GainsAndOffsetTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(41)

        cls.nwbfile_path = Path(mkdtemp()) / "test.nwb" 
        cls.nwbfile = NWBFile(session_description='test', identifier='test', session_start_time=datetime.now().astimezone())

        device = cls.nwbfile.create_device(name='recorder')
        electrode_group = cls.nwbfile.create_electrode_group('electrode', device=device, location='brain', description='fake')

        info = dict(group=electrode_group, location='brain')
        cls.number_of_electrodes = 10
        for id in range(cls.number_of_electrodes):
            cls.nwbfile.add_electrode(id=id, **info)

    def test_offset_extraction_from_electrode_table(self):
        
        offset_values = self.rng.integers(low=0, high=20, size=self.number_of_electrodes, ).astype('float')
        self.nwbfile.add_electrode_column(name="offset", data=offset_values, description="offset" )
        
        number_of_electrodes_in_electrical_series = 5
        options = range(self.number_of_electrodes)
        region_indexes = sorted(self.rng.choice(options, size=number_of_electrodes_in_electrical_series, replace=False).tolist())
        electrode_region = self.nwbfile.create_electrode_table_region(region_indexes, 'record electrodes')

        num_frames = 1_000
        data = self.rng.random((num_frames, number_of_electrodes_in_electrical_series))
        electrical_series_name = "test_electrical_series"
        rate = 30_000.0
        electrical_series = ElectricalSeries(name=electrical_series_name , data=data, electrodes=electrode_region, rate=rate)
        self.nwbfile.add_acquisition(electrical_series)
        
        with NWBHDF5IO(self.nwbfile_path, mode='w') as io:
            io.write(self.nwbfile)
            
        recording_extractor = NwbRecordingExtractor(self.nwbfile_path, electrical_series_name=electrical_series_name)
        extracted_offsets = recording_extractor.get_channel_offsets() / 1e6  # Conver to units 
        expected_offsets = offset_data[region_indexes]
        np.testing.assert_almost_equal(extracted_offsets, expected_offsets)

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries, mock_ElectrodeTable, mock_electrodes, mock_ElectrodeGroup

@pytest.fixture(scope="module")
def nwb_with_ecephys_content():
    nwbfile = mock_NWBFile()
    device = mock_Device(name="probe")
    nwbfile.add_device(device)
    nwbfile.add_electrode_column(name="channel_name", description="channel name")
    nwbfile.add_electrode_column(name="rel_x", description="rel_x")
    nwbfile.add_electrode_column(name="rel_y", description="rel_y")
    nwbfile.add_electrode_column(name="property", description="A property")

    electrode_group = mock_ElectrodeGroup(device=device)
    nwbfile.add_electrode_group(electrode_group)
    rel_x = 0.0
    property_value = "A"
    number_of_electrodes = 3
    electrodes_info = dict(group=electrode_group, location='brain', channel_name="channel_x", rel_x=rel_x, rel_y=0.0, property=property_value)
    electrode_indices = [0, 1, 2, 3, 4]

    for index in electrode_indices:
        electrodes_info["channel_name"] = f"{index}"
        electrodes_info["rel_y"] = float(index)
        nwbfile.add_electrode(id=index, **electrodes_info)


    electrical_series_electrodes = nwbfile.create_electrode_table_region(region=electrode_indices, description="electrodes for ElectricalSeries")
    electrical_series = mock_ElectricalSeries(name="ElectricalSeries1", electrodes=electrical_series_electrodes)
    nwbfile.add_acquisition(electrical_series)

    electrode_group = mock_ElectrodeGroup(device=device)
    nwbfile.add_electrode_group(electrode_group)

    rel_x = 3.0
    property_value = "B"
    electrode_indices = [5, 6, 7, 8, 9]

    electrodes_info = dict(group=electrode_group, location='brain', channel_name="channel_x", rel_x=rel_x, rel_y=0.0, property=property_value)

    for index in electrode_indices:
        electrodes_info["channel_name"] = f"{index}"
        electrodes_info["rel_y"] = float(index)
        nwbfile.add_electrode(id=index, **electrodes_info)


    electrical_series_electrodes = nwbfile.create_electrode_table_region(region=electrode_indices, description="electrodes for ElectricalSeries")
    electrical_series = mock_ElectricalSeries(name="ElectricalSeries2", electrodes=electrical_series_electrodes)
    nwbfile.add_acquisition(electrical_series)

    return nwbfile

@pytest.fixture(scope="module")
def path_to_nwbfile(nwb_with_ecephys_content, tmp_path_factory):

    nwbfile_path = tmp_path_factory.mktemp("nwb_tests_directory") / "test.nwb"
    print(nwbfile_path)
    print(nwb_with_ecephys_content)
    with NWBHDF5IO(nwbfile_path, mode='w') as io:
        io.write(nwb_with_ecephys_content)

    return nwbfile_path

def test_nwb_extractor_channel_retrieval(path_to_nwbfile):
    recording_extractor1 = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name="ElectricalSeries1")
    recording_extractor2 = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name="ElectricalSeries2")

    assert np.array_equal(recording_extractor1.channel_ids, ["0", "1", "2", "3", "4"])
    assert np.array_equal(recording_extractor2.channel_ids, ["5", "6", "7", "8", "9"])
    
   
if __name__ == '__main__':
    test = NwbRecordingTest()
    # ~ test = NwbSortingTest()

    test.setUp()
    test.test_open()
    
