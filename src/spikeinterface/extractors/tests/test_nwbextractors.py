import unittest
from pathlib import Path


import pytest
import numpy as np

from spikeinterface.extractors.extractor_classes import (
    NwbRecordingExtractor,
    NwbSortingExtractor,
    NwbTimeSeriesExtractor,
)

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite
from spikeinterface.core.testing import check_recordings_equal


class NwbRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbRecordingExtractor
    downloads = []
    entities = []


class NwbSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NwbSortingExtractor
    downloads = []
    entities = []


def nwbfile_with_ecephys_content():
    from pynwb.ecephys import ElectricalSeries, LFP, FilteredEphys
    from pynwb.testing.mock.file import mock_NWBFile
    from pynwb.testing.mock.device import mock_Device
    from pynwb.testing.mock.ecephys import mock_ElectricalSeries, mock_ElectrodeGroup

    to_micro_volts = 1e6

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
    offset = 1.0 * to_micro_volts
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
    channel_conversion = 2.0 * np.ones(len(electrode_indices)) * to_micro_volts
    electrical_series = mock_ElectricalSeries(
        name=electrical_series_name,
        electrodes=electrode_region,
        channel_conversion=channel_conversion,
    )
    nwbfile.add_acquisition(electrical_series)

    electrode_group = mock_ElectrodeGroup(device=device)
    nwbfile.add_electrode_group(electrode_group)

    rel_x = 3.0
    property_value = "B"
    offset = 2.0 * to_micro_volts
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
    conversion = 5.0 * to_micro_volts
    a_different_offset = offset + 1.0 * to_micro_volts
    electrical_series = ElectricalSeries(
        name=electrical_series_name,
        data=data,
        electrodes=electrode_region,
        rate=rate,
        offset=a_different_offset,
        conversion=conversion,
    )
    nwbfile.add_acquisition(electrical_series)

    # add electrical series in processing
    electrical_series_name = "ElectricalSeries1"
    electrode_indices = [5, 6, 7, 8, 9]
    data = rng.random(size=(num_frames, len(electrode_indices)))
    rate = 30_000.0
    conversion = 5.0 * to_micro_volts
    a_different_offset = offset + 1.0 * to_micro_volts
    electrical_series = ElectricalSeries(
        name=electrical_series_name,
        data=data,
        electrodes=electrode_region,
        rate=rate,
        conversion=conversion,
    )

    ecephys_mod = nwbfile.create_processing_module(name="ecephys", description="Ecephys module")
    ecephys_mod.add(LFP(name="LFP"))
    ecephys_mod.data_interfaces["LFP"].add_electrical_series(electrical_series)

    # custom module
    # add electrical series in custom preprocessing module
    electrical_series_name = "ElectricalSeries2"
    electrode_indices = [0, 1, 2, 3, 4]
    data = rng.random(size=(num_frames, len(electrode_indices)))
    rate = 30_000.0
    conversion = 5.0 * to_micro_volts
    a_different_offset = offset + 1.0 * to_micro_volts
    electrical_series = ElectricalSeries(
        name=electrical_series_name,
        data=data,
        electrodes=electrode_region,
        rate=rate,
        conversion=conversion,
    )

    custom_mod = nwbfile.create_processing_module(name="my_custom_module", description="Something custom")
    custom_mod.add(FilteredEphys(name="MyContainer"))
    custom_mod.data_interfaces["MyContainer"].add_electrical_series(electrical_series)

    return nwbfile


def _generate_nwbfile(backend, file_path):
    from pynwb import NWBHDF5IO
    from hdmf_zarr import NWBZarrIO

    nwbfile = nwbfile_with_ecephys_content()
    if backend == "hdf5":
        io_class = NWBHDF5IO
    elif backend == "zarr":
        io_class = NWBZarrIO
    with io_class(str(file_path), mode="w") as io:
        io.write(nwbfile)
    return file_path, nwbfile


@pytest.fixture(scope="module", params=["hdf5", "zarr"])
def generate_nwbfile(request, tmp_path_factory):
    nwbfile = nwbfile_with_ecephys_content()
    backend = request.param
    nwbfile_path = tmp_path_factory.mktemp("nwb_tests_directory") / "test.nwb"
    nwbfile_path, nwbfile = _generate_nwbfile(backend, nwbfile_path)
    return nwbfile_path, nwbfile


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_nwb_extractor_channel_ids_retrieval(generate_nwbfile, use_pynwb):
    """
    Test that the channel_ids are retrieved from the electrodes table ONLY from the corresponding
    region of the electrical series
    """
    path_to_nwbfile, nwbfile_with_ecephys_content = generate_nwbfile
    electrical_series_name_list = ["ElectricalSeries1", "ElectricalSeries2"]
    for electrical_series_name in electrical_series_name_list:
        recording_extractor = NwbRecordingExtractor(
            path_to_nwbfile,
            electrical_series_path=f"acquisition/{electrical_series_name}",
            use_pynwb=use_pynwb,
        )

        nwbfile = nwbfile_with_ecephys_content
        electrical_series = nwbfile.acquisition[electrical_series_name]
        electrical_series_electrode_indices = electrical_series.electrodes.data[:]
        electrodes_table = nwbfile.electrodes.to_dataframe()
        sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

        expected_channel_ids = sub_electrodes_table["channel_name"].values
        extracted_channel_ids = recording_extractor.channel_ids
        assert np.array_equal(extracted_channel_ids, expected_channel_ids)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_nwb_extractor_property_retrieval(generate_nwbfile, use_pynwb):
    """
    Test that the property is retrieved from the electrodes table ONLY from the corresponding
    region of the electrical series
    """
    path_to_nwbfile, nwbfile_with_ecephys_content = generate_nwbfile
    electrical_series_name_list = ["ElectricalSeries1", "ElectricalSeries2"]
    for electrical_series_name in electrical_series_name_list:
        recording_extractor = NwbRecordingExtractor(
            path_to_nwbfile,
            electrical_series_path=f"acquisition/{electrical_series_name}",
            use_pynwb=use_pynwb,
        )
        nwbfile = nwbfile_with_ecephys_content
        electrical_series = nwbfile.acquisition[electrical_series_name]
        electrical_series_electrode_indices = electrical_series.electrodes.data[:]
        electrodes_table = nwbfile.electrodes.to_dataframe()
        sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

        expected_property = sub_electrodes_table["property"].values
        extracted_property = recording_extractor.get_property("property")
        assert np.array_equal(extracted_property, expected_property)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_nwb_extractor_offset_from_electrodes_table(generate_nwbfile, use_pynwb):
    """Test that the offset is retrieved from the electrodes table if it is not present in the ElectricalSeries."""
    path_to_nwbfile, nwbfile_with_ecephys_content = generate_nwbfile

    electrical_series_name = "ElectricalSeries1"
    recording_extractor = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=f"acquisition/{electrical_series_name}",
        use_pynwb=use_pynwb,
    )
    nwbfile = nwbfile_with_ecephys_content
    electrical_series = nwbfile.acquisition[electrical_series_name]
    electrical_series_electrode_indices = electrical_series.electrodes.data[:]
    electrodes_table = nwbfile.electrodes.to_dataframe()
    sub_electrodes_table = electrodes_table.iloc[electrical_series_electrode_indices]

    expected_offsets_uV = sub_electrodes_table["offset"].values * 1e6
    extracted_offsets_uV = recording_extractor.get_channel_offsets()
    assert np.array_equal(extracted_offsets_uV, expected_offsets_uV)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_nwb_extractor_offset_from_series(generate_nwbfile, use_pynwb):
    """Test that the offset is retrieved from the ElectricalSeries if it is present."""
    path_to_nwbfile, nwbfile_with_ecephys_content = generate_nwbfile

    electrical_series_name = "ElectricalSeries2"
    recording_extractor = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=f"acquisition/{electrical_series_name}",
        use_pynwb=use_pynwb,
    )
    nwbfile = nwbfile_with_ecephys_content
    electrical_series = nwbfile.acquisition[electrical_series_name]
    expected_offsets_uV = electrical_series.offset * 1e6
    expected_offsets_uV = np.ones(recording_extractor.get_num_channels()) * expected_offsets_uV
    extracted_offsets_uV = recording_extractor.get_channel_offsets()
    assert np.array_equal(extracted_offsets_uV, expected_offsets_uV)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_retrieving_from_processing(generate_nwbfile, use_pynwb):
    """Test that the offset is retrieved from the ElectricalSeries if it is present."""
    path_to_nwbfile, nwbfile_with_ecephys_content = generate_nwbfile

    electrical_series_name = "ElectricalSeries1"
    module = "ecephys"
    data_interface = "LFP"
    recording_extractor_lfp = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=f"processing/{module}/{data_interface}/{electrical_series_name}",
        use_pynwb=use_pynwb,
    )
    nwbfile = nwbfile_with_ecephys_content
    electrical_series_lfp = nwbfile.processing[module].data_interfaces[data_interface][electrical_series_name]
    assert np.array_equal(electrical_series_lfp.data[:], recording_extractor_lfp.get_traces())

    electrical_series_name = "ElectricalSeries2"
    module = "my_custom_module"
    data_interface = "MyContainer"
    recording_extractor_custom = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=f"processing/{module}/{data_interface}/{electrical_series_name}",
        use_pynwb=use_pynwb,
    )
    nwbfile = nwbfile_with_ecephys_content
    electrical_series_custom = nwbfile.processing[module].data_interfaces[data_interface][electrical_series_name]
    assert np.array_equal(electrical_series_custom.data[:], recording_extractor_custom.get_traces())


def test_fetch_available_electrical_series_paths(generate_nwbfile):
    path_to_nwbfile, _ = generate_nwbfile
    available_electrical_series = NwbRecordingExtractor.fetch_available_electrical_series_paths(
        file_path=path_to_nwbfile
    )

    expected_paths = [
        "acquisition/ElectricalSeries1",
        "acquisition/ElectricalSeries2",
        "processing/ecephys/LFP/ElectricalSeries1",
        "processing/my_custom_module/MyContainer/ElectricalSeries2",
    ]

    assert available_electrical_series == expected_paths


@pytest.mark.parametrize("electrical_series_path", ["acquisition/ElectricalSeries1", "acquisition/ElectricalSeries2"])
def test_recording_equality_with_pynwb_and_backend(generate_nwbfile, electrical_series_path):
    path_to_nwbfile, _ = generate_nwbfile
    recording_extractor_backend = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=electrical_series_path,
        use_pynwb=False,
    )

    recording_extractor_pynwb = NwbRecordingExtractor(
        path_to_nwbfile,
        electrical_series_path=electrical_series_path,
        use_pynwb=True,
    )

    check_recordings_equal(recording_extractor_backend, recording_extractor_pynwb)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_failure_with_wrong_electrical_series_path(generate_nwbfile, use_pynwb):
    """Test that the extractor raises an error if the electrical series name is not found."""
    path_to_nwbfile, _ = generate_nwbfile
    with pytest.raises(ValueError):
        recording_extractor = NwbRecordingExtractor(
            path_to_nwbfile,
            electrical_series_path="acquisition/ElectricalSeries3",
            use_pynwb=use_pynwb,
        )


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_sorting_extraction_of_ragged_arrays(tmp_path, use_pynwb):
    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile

    nwbfile = mock_NWBFile()

    # Add the spikes
    nwbfile.add_unit_column(name="unit_name", description="the name of the unit")
    nwbfile.add_unit_column(name="a_property", description="a_cool_property")

    spike_times_a = np.array([0.0, 1.0, 2.0])
    nwbfile.add_unit(spike_times=spike_times_a, unit_name="a", a_property="a_property_value")
    spike_times_b = np.array([0.0, 1.0, 2.0, 3.0])
    nwbfile.add_unit(spike_times=spike_times_b, unit_name="b", a_property="b_property_value")

    non_uniform_ragged_array = [[1, 2, 3, 8, 10], [1, 2, 3, 5]]
    nwbfile.add_unit_column(
        name="non_uniform_ragged_array",
        description="A non-uniform ragged array that can not be loaded by spikeinterface",
        data=non_uniform_ragged_array,
        index=True,
    )

    uniform_ragged_array = [[1, 2, 3], [4, 5, 6]]
    nwbfile.add_unit_column(
        name="uniform_ragged_array",
        description="A uniform ragged array that can be loaded into spikeinterface",
        data=uniform_ragged_array,
        index=True,
    )

    doubled_ragged_array = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    nwbfile.add_unit_column(
        name="doubled_ragged_array",
        description="A doubled ragged array that can not be loaded into spikeinterface",
        data=doubled_ragged_array,
        index=2,
    )

    file_path = tmp_path / "test.nwb"
    # Write the nwbfile to a temporary file
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    sorting_extractor = NwbSortingExtractor(
        file_path=file_path,
        sampling_frequency=10.0,
        t_start=0,
        use_pynwb=use_pynwb,
    )

    units_ids = sorting_extractor.get_unit_ids()

    np.testing.assert_equal(units_ids, ["a", "b"])

    added_properties = sorting_extractor.get_property_keys()
    assert "non_uniform_ragged_array" not in added_properties
    assert "doubled_ragged_array" not in added_properties
    assert "uniform_ragged_array" in added_properties
    assert "a_property" in added_properties

    extracted_spike_times_a = sorting_extractor.get_unit_spike_train(unit_id="a", return_times=True)
    np.testing.assert_allclose(extracted_spike_times_a, spike_times_a)

    extracted_spike_times_b = sorting_extractor.get_unit_spike_train(unit_id="b", return_times=True)
    np.testing.assert_allclose(extracted_spike_times_b, spike_times_b)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_sorting_extraction_start_time(tmp_path, use_pynwb):

    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile

    nwbfile = mock_NWBFile()

    # Add the spikes

    t_start = 10
    sampling_frequency = 100.0
    spike_times0 = np.array([0.0, 1.0, 2.0]) + t_start
    nwbfile.add_unit(spike_times=spike_times0)
    spike_times1 = np.array([0.0, 1.0, 2.0, 3.0]) + t_start
    nwbfile.add_unit(spike_times=spike_times1)

    file_path = tmp_path / "test.nwb"
    # Write the nwbfile to a temporary file
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    sorting_extractor = NwbSortingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        t_start=t_start,
        use_pynwb=use_pynwb,
    )

    # Test frames
    extracted_frames0 = sorting_extractor.get_unit_spike_train(unit_id=0, return_times=False)
    expected_frames = ((spike_times0 - t_start) * sampling_frequency).astype("int64")
    np.testing.assert_allclose(extracted_frames0, expected_frames)

    extracted_frames1 = sorting_extractor.get_unit_spike_train(unit_id=1, return_times=False)
    expected_frames = ((spike_times1 - t_start) * sampling_frequency).astype("int64")
    np.testing.assert_allclose(extracted_frames1, expected_frames)

    # Test times
    extracted_spike_times0 = sorting_extractor.get_unit_spike_train(unit_id=0, return_times=True)
    expected_spike_times0 = spike_times0
    np.testing.assert_allclose(extracted_spike_times0, expected_spike_times0)

    extracted_spike_times1 = sorting_extractor.get_unit_spike_train(unit_id=1, return_times=True)
    expected_spike_times1 = spike_times1
    np.testing.assert_allclose(extracted_spike_times1, expected_spike_times1)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_sorting_extraction_start_time_from_series(tmp_path, use_pynwb):
    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile
    from pynwb.ecephys import ElectricalSeries, LFP, FilteredEphys

    from pynwb.testing.mock.ecephys import mock_electrodes

    nwbfile = mock_NWBFile()
    electrical_series_name = "ElectricalSeries"
    t_start = 10.0
    sampling_frequency = 100.0
    n_electrodes = 5
    electrodes = mock_electrodes(n_electrodes=n_electrodes, nwbfile=nwbfile)
    electrical_series = ElectricalSeries(
        name=electrical_series_name,
        starting_time=t_start,
        rate=sampling_frequency,
        data=np.ones((10, 5)),
        electrodes=electrodes,
    )
    nwbfile.add_acquisition(electrical_series)
    # Add the spikes
    spike_times0 = np.array([0.0, 1.0, 2.0]) + t_start
    nwbfile.add_unit(spike_times=spike_times0)
    spike_times1 = np.array([0.0, 1.0, 2.0, 3.0]) + t_start
    nwbfile.add_unit(spike_times=spike_times1)

    file_path = tmp_path / "test.nwb"
    # Write the nwbfile to a temporary file
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    sorting_extractor = NwbSortingExtractor(
        file_path=file_path,
        electrical_series_path=f"acquisition/{electrical_series_name}",
        use_pynwb=use_pynwb,
    )

    # Test frames
    extracted_frames0 = sorting_extractor.get_unit_spike_train(unit_id=0, return_times=False)
    expected_frames = ((spike_times0 - t_start) * sampling_frequency).astype("int64")
    np.testing.assert_allclose(extracted_frames0, expected_frames)

    extracted_frames1 = sorting_extractor.get_unit_spike_train(unit_id=1, return_times=False)
    expected_frames = ((spike_times1 - t_start) * sampling_frequency).astype("int64")
    np.testing.assert_allclose(extracted_frames1, expected_frames)

    # Test returned times
    extracted_spike_times0 = sorting_extractor.get_unit_spike_train(unit_id=0, return_times=True)
    expected_spike_times0 = spike_times0
    np.testing.assert_allclose(extracted_spike_times0, expected_spike_times0)

    extracted_spike_times1 = sorting_extractor.get_unit_spike_train(unit_id=1, return_times=True)
    expected_spike_times1 = spike_times1
    np.testing.assert_allclose(extracted_spike_times1, expected_spike_times1)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_get_unit_spike_train_in_seconds(tmp_path, use_pynwb):
    """Test that get_unit_spike_train_in_seconds returns accurate timestamps without double conversion."""
    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile

    nwbfile = mock_NWBFile()

    # Add units with known spike times
    t_start = 5.0
    sampling_frequency = 1000.0
    spike_times_unit_a = np.array([5.1, 5.2, 5.3, 6.0, 6.5])  # Absolute times
    spike_times_unit_b = np.array([5.05, 5.15, 5.25, 5.35, 6.1])  # Absolute times

    nwbfile.add_unit(spike_times=spike_times_unit_a)
    nwbfile.add_unit(spike_times=spike_times_unit_b)

    file_path = tmp_path / "test.nwb"
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    sorting_extractor = NwbSortingExtractor(
        file_path=file_path,
        sampling_frequency=sampling_frequency,
        t_start=t_start,
        use_pynwb=use_pynwb,
    )

    # Test full spike trains
    spike_times_a_direct = sorting_extractor.get_unit_spike_train_in_seconds(unit_id=0)
    spike_times_a_legacy = sorting_extractor.get_unit_spike_train(unit_id=0, return_times=True)

    spike_times_b_direct = sorting_extractor.get_unit_spike_train_in_seconds(unit_id=1)
    spike_times_b_legacy = sorting_extractor.get_unit_spike_train(unit_id=1, return_times=True)

    # Both methods should return exact timestamps since return_times now uses get_unit_spike_train_in_seconds
    np.testing.assert_array_equal(spike_times_a_direct, spike_times_unit_a)
    np.testing.assert_array_equal(spike_times_b_direct, spike_times_unit_b)
    np.testing.assert_array_equal(spike_times_a_legacy, spike_times_unit_a)
    np.testing.assert_array_equal(spike_times_b_legacy, spike_times_unit_b)

    # Test time filtering
    start_time = 5.2
    end_time = 6.1

    # Direct method with time filtering
    spike_times_a_filtered = sorting_extractor.get_unit_spike_train_in_seconds(
        unit_id=0, start_time=start_time, end_time=end_time
    )
    spike_times_b_filtered = sorting_extractor.get_unit_spike_train_in_seconds(
        unit_id=1, start_time=start_time, end_time=end_time
    )

    # Expected filtered results
    expected_a = spike_times_unit_a[(spike_times_unit_a >= start_time) & (spike_times_unit_a < end_time)]
    expected_b = spike_times_unit_b[(spike_times_unit_b >= start_time) & (spike_times_unit_b < end_time)]

    np.testing.assert_array_equal(spike_times_a_filtered, expected_a)
    np.testing.assert_array_equal(spike_times_b_filtered, expected_b)

    # Test edge cases
    # Start time filtering only
    spike_times_from_start = sorting_extractor.get_unit_spike_train_in_seconds(unit_id=0, start_time=5.25)
    expected_from_start = spike_times_unit_a[spike_times_unit_a >= 5.25]
    np.testing.assert_array_equal(spike_times_from_start, expected_from_start)

    # End time filtering only
    spike_times_to_end = sorting_extractor.get_unit_spike_train_in_seconds(unit_id=0, end_time=6.0)
    expected_to_end = spike_times_unit_a[spike_times_unit_a < 6.0]
    np.testing.assert_array_equal(spike_times_to_end, expected_to_end)

    # Test that direct method avoids frame conversion rounding errors
    # by comparing exact values that would be lost in frame conversion
    precise_times = np.array([5.1001, 5.1002, 5.1003])
    nwbfile_precise = mock_NWBFile()
    nwbfile_precise.add_unit(spike_times=precise_times)

    file_path_precise = tmp_path / "test_precise.nwb"
    with NWBHDF5IO(path=file_path_precise, mode="w") as io:
        io.write(nwbfile_precise)

    sorting_precise = NwbSortingExtractor(
        file_path=file_path_precise,
        sampling_frequency=sampling_frequency,
        t_start=t_start,
        use_pynwb=use_pynwb,
    )

    # Direct method should preserve exact precision
    direct_precise = sorting_precise.get_unit_spike_train_in_seconds(unit_id=0)
    np.testing.assert_array_equal(direct_precise, precise_times)

    # Both methods should now preserve exact precision since return_times uses get_unit_spike_train_in_seconds
    legacy_precise = sorting_precise.get_unit_spike_train(unit_id=0, return_times=True)
    # Both methods should be exactly equal since return_times now avoids double conversion
    np.testing.assert_array_equal(direct_precise, precise_times)
    np.testing.assert_array_equal(legacy_precise, precise_times)
    # Verify both methods return identical results
    np.testing.assert_array_equal(direct_precise, legacy_precise)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_multiple_unit_tables(tmp_path, use_pynwb):
    from pynwb.misc import Units
    from pynwb import NWBHDF5IO
    from pynwb.testing.mock.file import mock_NWBFile

    nwbfile = mock_NWBFile()

    # Add the spikes to the first unit table
    nwbfile.add_unit_column(name="unit_name", description="the name of the unit")
    nwbfile.add_unit_column(name="a_property", description="a_cool_property")
    spike_times_a = np.array([0.0, 1.0, 2.0])
    nwbfile.add_unit(spike_times=spike_times_a, unit_name="a", a_property="a_property_value")
    spike_times_b = np.array([0.0, 1.0, 2.0, 3.0])
    nwbfile.add_unit(spike_times=spike_times_b, unit_name="b", a_property="b_property_value")

    # Add the spikes to the second unit tabl

    # Add a second unit table to first NWBFile
    second_units_table = Units(
        name="units_raw",
        description="test units table",
        columns=[
            dict(name="unit_name", description="unit name"),
            dict(name="a_second_property", description="test property"),
        ],
    )
    spike_times_a1 = np.array([0.0, 1.0, 2.0, 3.0])
    second_units_table.add_unit(spike_times=spike_times_a1, unit_name="a1", a_second_property="a1_property_value")
    spike_times_b1 = np.array([0.0, 1.0, 2.0])
    second_units_table.add_unit(spike_times=spike_times_b1, unit_name="b1", a_second_property="b1_property_value")
    processing = nwbfile.create_processing_module(name="ecephys", description="test processing module")
    processing.add(second_units_table)

    file_path = tmp_path / "test.nwb"
    # Write the nwbfile to a temporary file
    with NWBHDF5IO(path=file_path, mode="w") as io:
        io.write(nwbfile)

    # passing a non existing unit table name should raise an error
    with pytest.raises(ValueError):
        sorting_extractor = NwbSortingExtractor(
            file_path=file_path, sampling_frequency=10.0, t_start=0, use_pynwb=use_pynwb, unit_table_path="units2"
        )

    sorting_extractor_main = NwbSortingExtractor(
        file_path=file_path,
        sampling_frequency=10.0,
        t_start=0,
        use_pynwb=use_pynwb,
        unit_table_path="units",
    )
    assert np.array_equal(sorting_extractor_main.unit_ids, ["a", "b"])
    assert "a_property" in sorting_extractor_main.get_property_keys()
    assert "a_second_property" not in sorting_extractor_main.get_property_keys()

    sorting_extractor_processing = NwbSortingExtractor(
        file_path=file_path,
        sampling_frequency=10.0,
        t_start=0,
        use_pynwb=use_pynwb,
        unit_table_path="processing/ecephys/units_raw",
    )
    assert np.array_equal(sorting_extractor_processing.unit_ids, ["a1", "b1"])
    assert "a_property" not in sorting_extractor_processing.get_property_keys()
    assert "a_second_property" in sorting_extractor_processing.get_property_keys()


def nwbfile_with_timeseries():
    from pynwb.testing.mock.file import mock_NWBFile
    from pynwb.base import TimeSeries

    nwbfile = mock_NWBFile()

    # Add regular TimeSeries with rate
    num_frames = 10
    num_channels = 5
    rng = np.random.default_rng(0)
    data = rng.random(size=(num_frames, num_channels))
    rate = 30_000.0
    starting_time = 0.0

    timeseries = TimeSeries(name="TimeSeries", data=data, rate=rate, starting_time=starting_time, unit="volts")
    nwbfile.add_acquisition(timeseries)

    # Add TimeSeries with timestamps
    timestamps = np.arange(num_frames) / rate
    timestamps[2] = 0
    timeseries_with_timestamps = TimeSeries(
        name="TimeSeriesWithTimestamps", data=data, timestamps=timestamps, unit="volts"
    )
    nwbfile.add_acquisition(timeseries_with_timestamps)

    # Add single channel TimeSeries
    single_channel_data = rng.random(size=(num_frames,))
    single_channel_series = TimeSeries(name="SingleChannelSeries", data=single_channel_data, rate=rate, unit="volts")
    nwbfile.add_acquisition(single_channel_series)

    # Add TimeSeries in processing module
    processing = nwbfile.create_processing_module(name="test_module", description="test module")
    proc_timeseries = TimeSeries(name="ProcessingTimeSeries", data=data, rate=rate, unit="volts")
    processing.add(proc_timeseries)

    return nwbfile


def _generate_nwbfile_with_time_series(backend, file_path):
    from pynwb import NWBHDF5IO
    from hdmf_zarr import NWBZarrIO

    nwbfile = nwbfile_with_timeseries()
    if backend == "hdf5":
        io_class = NWBHDF5IO
    elif backend == "zarr":
        io_class = NWBZarrIO
    with io_class(str(file_path), mode="w") as io:
        io.write(nwbfile)
    return file_path, nwbfile


@pytest.fixture(scope="module", params=["hdf5", "zarr"])
def generate_nwbfile_with_time_series(request, tmp_path_factory):
    backend = request.param
    nwbfile_path = tmp_path_factory.mktemp("nwb_tests_directory") / "test.nwb"
    nwbfile_path, nwbfile = _generate_nwbfile_with_time_series(backend, nwbfile_path)
    return nwbfile_path, nwbfile


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_timeseries_basic_functionality(generate_nwbfile_with_time_series, use_pynwb):
    """Test basic functionality with a regular TimeSeries."""
    path_to_nwbfile, nwbfile = generate_nwbfile_with_time_series

    recording = NwbTimeSeriesExtractor(path_to_nwbfile, timeseries_path="acquisition/TimeSeries", use_pynwb=use_pynwb)

    timeseries = nwbfile.acquisition["TimeSeries"]

    # Check data matches
    assert np.array_equal(recording.get_traces(), timeseries.data[:])

    # Check sampling frequency matches
    assert recording.get_sampling_frequency() == timeseries.rate

    # Check number of channels matches
    assert recording.get_num_channels() == timeseries.data.shape[1]


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_timeseries_with_timestamps(generate_nwbfile_with_time_series, use_pynwb):
    """Test functionality with a TimeSeries using timestamps."""
    path_to_nwbfile, nwbfile = generate_nwbfile_with_time_series

    recording = NwbTimeSeriesExtractor(
        path_to_nwbfile, timeseries_path="acquisition/TimeSeriesWithTimestamps", use_pynwb=use_pynwb
    )

    timeseries = nwbfile.acquisition["TimeSeriesWithTimestamps"]

    # Check data matches
    assert np.array_equal(recording.get_traces(), timeseries.data[:])

    # Check sampling frequency is correctly estimated
    expected_sampling_frequency = 1.0 / np.median(np.diff(timeseries.timestamps[:1000]))
    assert np.isclose(recording.get_sampling_frequency(), expected_sampling_frequency)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_time_series_load_time_vector(generate_nwbfile_with_time_series, use_pynwb):
    """Test loading time vector from TimeSeries with timestamps."""
    path_to_nwbfile, nwbfile = generate_nwbfile_with_time_series

    recording = NwbTimeSeriesExtractor(
        path_to_nwbfile,
        timeseries_path="acquisition/TimeSeriesWithTimestamps",
        load_time_vector=True,
        use_pynwb=use_pynwb,
    )

    timeseries = nwbfile.acquisition["TimeSeriesWithTimestamps"]

    times = recording.get_times()

    np.testing.assert_almost_equal(times, timeseries.timestamps[:])


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_single_channel_timeseries(generate_nwbfile_with_time_series, use_pynwb):
    """Test functionality with a single channel TimeSeries."""
    path_to_nwbfile, nwbfile = generate_nwbfile_with_time_series

    recording = NwbTimeSeriesExtractor(
        path_to_nwbfile, timeseries_path="acquisition/SingleChannelSeries", use_pynwb=use_pynwb
    )

    timeseries = nwbfile.acquisition["SingleChannelSeries"]

    # Check data matches
    assert np.array_equal(recording.get_traces().squeeze(), timeseries.data[:])

    # Check it's treated as a single channel
    assert recording.get_num_channels() == 1


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_processing_module_timeseries(generate_nwbfile_with_time_series, use_pynwb):
    """Test accessing TimeSeries from a processing module."""
    path_to_nwbfile, nwbfile = generate_nwbfile_with_time_series

    recording = NwbTimeSeriesExtractor(
        path_to_nwbfile, timeseries_path="processing/test_module/ProcessingTimeSeries", use_pynwb=use_pynwb
    )

    timeseries = nwbfile.processing["test_module"]["ProcessingTimeSeries"]

    # Check data matches
    assert np.array_equal(recording.get_traces(), timeseries.data[:])


def test_fetch_available_timeseries_paths(generate_nwbfile_with_time_series):
    """Test the fetch_available_timeseries_paths static method."""
    path_to_nwbfile, _ = generate_nwbfile_with_time_series

    available_timeseries = NwbTimeSeriesExtractor.fetch_available_timeseries_paths(file_path=path_to_nwbfile)

    expected_paths = [
        "acquisition/TimeSeries",
        "acquisition/TimeSeriesWithTimestamps",
        "acquisition/SingleChannelSeries",
        "processing/test_module/ProcessingTimeSeries",
    ]

    assert sorted(available_timeseries) == sorted(expected_paths)


@pytest.mark.parametrize("use_pynwb", [True, False])
def test_error_with_wrong_timeseries_path(generate_nwbfile_with_time_series, use_pynwb):
    """Test that appropriate error is raised for non-existent TimeSeries."""
    path_to_nwbfile, _ = generate_nwbfile_with_time_series

    with pytest.raises(ValueError):
        _ = NwbTimeSeriesExtractor(
            path_to_nwbfile, timeseries_path="acquisition/NonExistentTimeSeries", use_pynwb=use_pynwb
        )


def test_time_series_recording_equality_with_pynwb_and_backend(generate_nwbfile_with_time_series):
    """Test that pynwb and backend (h5py/zarr) modes produce identical results."""
    path_to_nwbfile, _ = generate_nwbfile_with_time_series

    recording_backend = NwbTimeSeriesExtractor(
        path_to_nwbfile, timeseries_path="acquisition/TimeSeries", use_pynwb=False
    )

    recording_pynwb = NwbTimeSeriesExtractor(path_to_nwbfile, timeseries_path="acquisition/TimeSeries", use_pynwb=True)

    check_recordings_equal(recording_backend, recording_pynwb)


if __name__ == "__main__":
    tmp_path = Path("tmp")
    if tmp_path.is_dir():
        import shutil

        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    use_pynwb = True
    gen = _generate_nwbfile("hdf5", tmp_path / "test.nwb")
    test_sorting_extraction_of_ragged_arrays(tmp_path, use_pynwb)
