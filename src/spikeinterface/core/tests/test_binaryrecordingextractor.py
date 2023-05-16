import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import BinaryRecordingExtractor


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BinaryRecordingExtractor():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'

    file_paths = [
        cache_folder / f'test_BinaryRecordingExtractor_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        np.memmap(file_paths[i], dtype=dtype, mode='w+',
                  shape=(num_samples, num_chan))

    rec = BinaryRecordingExtractor(
        file_paths, sampling_frequency, num_chan, dtype)
    print(rec)

    file_paths = [
        cache_folder / f'test_BinaryRecordingExtractor_copied_{i}.raw' for i in range(num_seg)]
    BinaryRecordingExtractor.write_recording(rec, file_paths)

    file_paths = [
        cache_folder / f'test_BinaryRecordingExtractor_{i}.raw' for i in range(num_seg)]
    assert (cache_folder / 'test_BinaryRecordingExtractor_copied_0.raw').is_file()


if __name__ == '__main__':
    test_BinaryRecordingExtractor()
