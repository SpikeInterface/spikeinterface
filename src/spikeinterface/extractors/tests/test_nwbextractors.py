import unittest
import unittest
from pathlib import Path
from tempfile import mkdtemp
from datetime import datetime

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
    
    def setUp(self):
        self.nwbfile_path = Path(mkdtemp()) / "test.nwb" 
        self.nwbfile = NWBFile(session_description='test', identifier='test', session_start_time=datetime.now().astimezone())

        device = self.nwbfile.create_device(name='recorder')
        electrode_group = self.nwbfile.create_electrode_group('electrode', device=device, location='brain', description='fake')
        info = dict(group=electrode_group, location='brain')

        #nwbfile.add_electrode_column(name='offset', index=False, description='Level offset')
        self.rng = np.random.default_rng(41)
        self.number_of_electrodes = 10
        for id in range(self.number_of_electrodes):
            self.nwbfile.add_electrode(id=id, **info)

    def test_offset_extraction_from_electrode_table(self):
        offset_data = self.rng.integers(low=0, high=20, size=self.number_of_electrodes).astype('float')
        self.nwbfile.add_electrode_column(name="offset", data=offset_data, description="Level offset" )
        number_of_electrodes_in_electrical_series = 5
        options = range(self.number_of_electrodes)
        region_indexes = sorted(self.rng.choice(options, size=number_of_electrodes_in_electrical_series, replace=False).tolist())
        electrode_region = self.nwbfile.create_electrode_table_region(region_indexes, 'record electrodes')

        num_frames = 10_000
        data = self.rng.random((num_frames, number_of_electrodes_in_electrical_series))
        electrical_series_name = "test_electrical_series"
        electrical_series = ElectricalSeries(name=electrical_series_name , data=data, electrodes=electrode_region, rate=20_000.0)

        self.nwbfile.add_acquisition(electrical_series)
        with NWBHDF5IO(self.nwbfile_path, mode='w') as io:
            io.write(self.nwbfile)
            
        recording_extractor = NwbRecordingExtractor(self.nwbfile_path, electrical_series_name=electrical_series_name)
        extracted_offsets = recording_extractor.get_channel_offsets() / 1e6  # Conver to units 
        expected_offsets = offset_data[region_indexes]
        np.testing.assert_almost_equal(extracted_offsets, expected_offsets)
                        
if __name__ == '__main__':
    test = NwbRecordingTest()
    # ~ test = NwbSortingTest()

    test.setUp()
    test.test_open()
    
