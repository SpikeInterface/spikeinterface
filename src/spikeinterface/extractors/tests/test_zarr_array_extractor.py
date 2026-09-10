import numpy as np
import pytest
import zarr

from spikeinterface.core.zarrextractors import read_zarr_array


@pytest.fixture
def make_dummy_zarr_data(tmp_path):
    """Create 2D Zarr array on disk."""
    zarr_path = tmp_path / "test_traces.zarr"
    num_samples = 1000
    num_channels = 4
    dtype = np.int16

    # dummy data
    original_data = (np.arange(num_samples * num_channels, dtype=dtype) % 500).reshape(num_samples, num_channels)

    z = zarr.open(
        store=str(zarr_path),
        mode="w",
        shape=original_data.shape,
        chunks=(200, num_channels),
        dtype=dtype,
    )
    z[:] = original_data

    return zarr_path, original_data


def test_zarr_array_extractor(make_dummy_zarr_data):
    """
    Make a zarr array then reads it using `read_zarr_array` and
    checks basic properties.
    """
    zarr_path, original_data = make_dummy_zarr_data
    sampling_frequency = 30000.0
    gain = 0.195
    offset = 10.0

    rec = read_zarr_array(
        file_path=zarr_path,
        sampling_frequency=sampling_frequency,
        gain_to_uV=gain,
        offset_to_uV=offset,
        is_filtered=False,
    )

    assert rec.get_num_channels() == original_data.shape[1]
    assert rec.get_num_samples() == original_data.shape[0]
    assert rec.get_sampling_frequency() == sampling_frequency
    assert np.all(rec.get_channel_gains() == gain)
    assert np.all(rec.get_channel_offsets() == offset)

    # 3. Verify exact trace reading
    traces_raw = rec.get_traces(return_scaled=False)
    np.testing.assert_array_equal(traces_raw, original_data)

    # 4. Verify channel and time slicing
    subset = rec.get_traces(start_frame=50, end_frame=150, channel_ids=[0, 2], return_in_uV=False)
    np.testing.assert_array_equal(subset, original_data[50:150, [0, 2]])

    # 5. Verify scaling math
    traces_scaled = rec.get_traces(start_frame=0, end_frame=10, channel_ids=[1], return_in_uV=True)
    expected_scaled = original_data[0:10, [1]] * gain + offset
    np.testing.assert_allclose(traces_scaled, expected_scaled, rtol=1e-5)
