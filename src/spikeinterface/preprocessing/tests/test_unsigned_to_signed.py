import numpy as np

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import unsigned_to_signed


def test_unsigned_to_signed():
    rng = np.random.RandomState(0)
    traces = rng.rand(10000, 4) * 100 + 2**15
    traces_uint16 = traces.astype("uint16")
    traces = rng.rand(10000, 4) * 100 + 2**31
    traces_uint32 = traces.astype("uint32")
    traces = rng.rand(10000, 4) * 100 + 2**11
    traces_uint16_12bits = traces.astype("uint16")
    rec_uint16 = NumpyRecording(traces_uint16, sampling_frequency=30000)
    rec_uint32 = NumpyRecording(traces_uint32, sampling_frequency=30000)
    rec_uint16_12bits = NumpyRecording(traces_uint16_12bits, sampling_frequency=30000)

    traces_int16 = (traces_uint16.astype("int32") - 2**15).astype("int16")
    np.testing.assert_array_equal(traces_int16, unsigned_to_signed(rec_uint16).get_traces())
    traces_int32 = (traces_uint32.astype("int64") - 2**31).astype("int32")
    np.testing.assert_array_equal(traces_int32, unsigned_to_signed(rec_uint32).get_traces())
    traces_int16_12bits = (traces_uint16_12bits.astype("int32") - 2**11).astype("int16")
    np.testing.assert_array_equal(traces_int16_12bits, unsigned_to_signed(rec_uint16_12bits, bit_depth=12).get_traces())


if __name__ == "__main__":
    test_unsigned_to_signed()
