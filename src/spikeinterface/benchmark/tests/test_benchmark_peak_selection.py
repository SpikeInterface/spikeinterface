import pytest

from pathlib import Path

@pytest.mark.skip()
def test_benchmark_peak_selection(create_cache_folder):
    cache_folder = create_cache_folder


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks"
    test_benchmark_peak_selection(cache_folder)
