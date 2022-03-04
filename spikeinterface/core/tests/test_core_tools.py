import platform
import pytest
from pathlib import Path

from spikeinterface.core.testing_tools import generate_recording

from spikeinterface.core.core_tools import write_binary_recording, write_memory_recording

try:
    from multiprocessing.shared_memory import SharedMemory

    HAVE_SHAREDMEMORY = True
except:
    HAVE_SHAREDMEMORY = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_write_binary_recording():
    # 2 segments
    recording = generate_recording(num_channels=2, durations=[10.325, 3.5])
    # make dumpable
    recording = recording.save()

    # write with loop
    write_binary_recording(recording, file_paths=[cache_folder / 'binary01.raw', cache_folder / 'binary02.raw'],
                           dtype=None, verbose=True, n_jobs=1)

    write_binary_recording(recording, file_paths=[cache_folder / 'binary01.raw', cache_folder / 'binary02.raw'],
                           dtype=None, verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)

    # write parrallel
    write_binary_recording(recording, file_paths=[cache_folder / 'binary01.raw', cache_folder / 'binary02.raw'],
                           dtype=None, verbose=False, n_jobs=2, chunk_memory='100k')

    # write parrallel
    write_binary_recording(recording, file_paths=[cache_folder / 'binary01.raw', cache_folder / 'binary02.raw'],
                           dtype=None, verbose=False, n_jobs=2, total_memory='200k', progress_bar=True)


def test_write_memory_recording():
    # 2 segments
    recording = generate_recording(num_channels=2, durations=[10.325, 3.5])
    # make dumpable
    recording = recording.save()

    # write with loop
    write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1)

    write_memory_recording(recording, dtype=None,
                           verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)

    if HAVE_SHAREDMEMORY and platform.system() != 'Windows':
        # write parrallel
        write_memory_recording(recording, dtype=None,
                               verbose=False, n_jobs=2, chunk_memory='100k')

        # write parrallel
        write_memory_recording(recording, dtype=None,
                               verbose=False, n_jobs=2, total_memory='200k', progress_bar=True)


if __name__ == '__main__':
    # test_write_binary_recording()
    test_write_memory_recording()
