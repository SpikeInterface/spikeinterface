import pytest
from pathlib import Path

from spikeinterface.core import NpySnippetsExtractor
from spikeinterface.core import generate_snippets

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_NpySnippetsExtractor():
    segment_durations = [2,5]
    sampling_frequency = 30000
    file_path = [cache_folder / f'test_NpySnippetsExtractor_{i}.npy' for i in range(len(segment_durations))]
    
    snippets, _ = generate_snippets(sampling_frequency=sampling_frequency,
                                durations=segment_durations)

    NpySnippetsExtractor.write_snippets(snippets,file_path)
    npy_snippets = NpySnippetsExtractor(file_path, sampling_frequency=sampling_frequency)

    assert npy_snippets.get_num_segments() == snippets.get_num_segments()
    assert npy_snippets.get_num_snippets(1) == snippets.get_num_snippets(1)
    assert npy_snippets.snippet_len == snippets.snippet_len


if __name__ == '__main__':
    test_NpySnippetsExtractor()
