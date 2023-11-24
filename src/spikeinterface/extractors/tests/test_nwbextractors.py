import unittest
import unittest
from pathlib import Path
from tempfile import mkdtemp
from datetime import datetime

import pytest
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectricalSeries, mock_ElectrodeGroup

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


from pynwb.testing.mock.ecephys import mock_ElectrodeGroup


@pytest.fixture(scope="module")
def nwbfile_with_ecephys_content():
    nwbfile = mock_NWBFile()
    device = mock_Device(name="probe")
    nwbfile.add_device(device)
    nwbfile.add_electrode_column(name="channel_name", description="channel name")
    nwbfile.add_electrode_column(name="rel_x", description="rel_x")
    nwbfile.add_electrode_column(name="rel_y", description="rel_y")
    nwbfile.add_electrode_column(name="property", description="A property")
    nwbfile.add_electrode_column(name="electrical_series_name", description="Electrical series name")
    nwbfile.add_electrode_column(name="offset", description="Electrical series offset")

    electrode_group = mock_ElectrodeGroup(device=device)
    nwbfile.add_electrode_group(electrode_group)
    rel_x = 0.0
    property_value = "A"
    offset = 1.0
    electrical_series_name = "ElectricalSeries1"
    electrode_indices = [0, 1, 2, 3, 4]

    electrodes_info = dict(
        group=electrode_group,
        location="brain",
        rel_x=rel_x,
        property=property_value,
        electrical_series_name=electrical_series_name,
        offset=offset,
    )

    for index in electrode_indices:
        electrodes_info["channel_name"] = f"{index}"
        electrodes_info["rel_y"] = float(index)
        nwbfile.add_electrode(id=index, **electrodes_info)

    electrode_region = nwbfile.create_electrode_table_region(region=electrode_indices, description="electrodes")
    electrical_series = mock_ElectricalSeries(name=electrical_series_name, electrodes=electrode_region)
    nwbfile.add_acquisition(electrical_series)

    electrode_group = mock_ElectrodeGroup(device=device)
    nwbfile.add_electrode_group(electrode_group)

    rel_x = 3.0
    property_value = "B"
    offset = 2.0
    electrical_series_name = "ElectricalSeries2"
    electrode_indices = [5, 6, 7, 8, 9]

    electrodes_info = dict(
        group=electrode_group,
        location="brain",
        rel_x=rel_x,
        property=property_value,
        electrical_series_name=electrical_series_name,
        offset=offset,
    )

    for index in electrode_indices:
        electrodes_info["channel_name"] = f"{index}"
        electrodes_info["rel_y"] = float(index)
        nwbfile.add_electrode(id=index, **electrodes_info)

    electrode_region = nwbfile.create_electrode_table_region(region=electrode_indices, description="electrodes")
    num_frames = 1_000
    rng = np.random.default_rng(0)
    data = rng.random(size=(num_frames, len(electrode_indices)))
    rate = 30_000.0
    electrical_series = ElectricalSeries(
        name=electrical_series_name, data=data, electrodes=electrode_region, rate=rate, offset=offset + 1.0
    )
    nwbfile.add_acquisition(electrical_series)

    return nwbfile


@pytest.fixture(scope="module")
def path_to_nwbfile(nwbfile_with_ecephys_content, tmp_path_factory):
    nwbfile_path = tmp_path_factory.mktemp("nwb_tests_directory") / "test.nwb"
    with NWBHDF5IO(nwbfile_path, mode="w") as io:
        io.write(nwbfile_with_ecephys_content)

    return nwbfile_path


def test_nwb_extractor_channel_ids_retrieval(path_to_nwbfile, nwbfile_with_ecephys_content):
    """
    Test that the channel_ids are retrieved from the electrodes table ONLY from the corresponding
    region of the electrical series
    """
    electrical_series_name_list = ["ElectricalSeries1", "ElectricalSeries2"]
    for electrical_series_name in electrical_series_name_list:
        recording_extractor = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name=electrical_series_name)

        nwbfile = nwbfile_with_ecephys_content
        electrical_series = nwbfile.acquisition[electrical_series_name]
        electrical_series_electrode_indices = electrical_series.electrodes.data[:]
        electrodes_table = nwbfile.electrodes.to_dataframe()
        sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

        expected_channel_ids = sub_electrodes_table["channel_name"].values
        extracted_channel_ids = recording_extractor.channel_ids
        assert np.array_equal(extracted_channel_ids, expected_channel_ids)


def test_nwb_extractor_property_retrieval(path_to_nwbfile, nwbfile_with_ecephys_content):
    """
    Test that the property is retrieved from the electrodes table ONLY from the corresponding
    region of the electrical series
    """

    electrical_series_name_list = ["ElectricalSeries1", "ElectricalSeries2"]
    for electrical_series_name in electrical_series_name_list:
        recording_extractor = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name=electrical_series_name)

        nwbfile = nwbfile_with_ecephys_content
        electrical_series = nwbfile.acquisition[electrical_series_name]
        electrical_series_electrode_indices = electrical_series.electrodes.data[:]
        electrodes_table = nwbfile.electrodes.to_dataframe()
        sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

        expected_property = sub_electrodes_table["property"].values
        extracted_property = recording_extractor.get_property("property")
        assert np.array_equal(extracted_property, expected_property)


def test_nwb_extractor_offset_from_electrodes_table(path_to_nwbfile, nwbfile_with_ecephys_content):
    """Test that the offset is retrieved from the electrodes table if it is not present in the ElectricalSeries."""
    electrical_series_name = "ElectricalSeries1"
    recording_extractor = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name=electrical_series_name)

    nwbfile = nwbfile_with_ecephys_content
    electrical_series = nwbfile.acquisition[electrical_series_name]
    electrical_series_electrode_indices = electrical_series.electrodes.data[:]
    electrodes_table = nwbfile.electrodes.to_dataframe()
    sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

    expected_offsets_uV = sub_electrodes_table["offset"].values * 1e6
    extracted_offsets_uV = recording_extractor.get_channel_offsets()
    assert np.array_equal(extracted_offsets_uV, expected_offsets_uV)


def test_nwb_extractor_offset_from_series(path_to_nwbfile, nwbfile_with_ecephys_content):
    """Test that the offset is retrieved from the ElectricalSeries if it is present."""
    electrical_series_name = "ElectricalSeries2"
    recording_extractor = NwbRecordingExtractor(path_to_nwbfile, electrical_series_name=electrical_series_name)

    nwbfile = nwbfile_with_ecephys_content
    electrical_series = nwbfile.acquisition[electrical_series_name]
    expected_offsets_uV = electrical_series.offset * 1e6
    expected_offsets_uV = np.ones(recording_extractor.get_num_channels()) * expected_offsets_uV
    extracted_offsets_uV = recording_extractor.get_channel_offsets()
    assert np.array_equal(extracted_offsets_uV, expected_offsets_uV)


def test_sorting_extraction_of_ragged_arrays(tmp_path):
    nwbfile = mock_NWBFile()

    # Add the spikes
    nwbfile.add_unit_column(name="unit_name", description="the name of the unit")
    spike_times1 = np.array([0.0, 1.0, 2.0])
    nwbfile.add_unit(spike_times=spike_times1, unit_name="a")
    spike_times2 = np.array([0.0, 1.0, 2.0, 3.0])
    nwbfile.add_unit(spike_times=spike_times2, unit_name="b")

    ragged_array_bad = [[1, 2, 3], [1, 2, 3, 5]]
    nwbfile.add_unit_column(
        name="ragged_array_bad",
        description="an evill array that wants to destroy your test",
        data=ragged_array_bad,
        index=True,
    )

    ragged_array_good = [[1, 2], [3, 4]]
    nwbfile.add_unit_column(
        name="ragged_array_good",
        description="a good array that wants to help your test be nice to nice arrays",
        data=ragged_array_good,
        index=True,
    )

    file_path = tmp_path / "test.nwb"
    # Write the nwbfile to a temporary file
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    sorting_extractor = NwbSortingExtractor(file_path=file_path, sampling_frequency=10.0)

    # Check that the bad array was not added
    added_properties = sorting_extractor.get_property_keys()
    assert "ragged_array_bad" not in added_properties
    assert "ragged_array_good" in added_properties


if __name__ == "__main__":
    test = NwbRecordingTest()
