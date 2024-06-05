import pytest


@pytest.fixture(scope="module")
def create_cache_folder(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    return cache_folder


@pytest.mark.skip()
def test_benchmark_peak_selection(create_cache_folder):
    cache_folder = create_cache_folder
    pass


if __name__ == "__main__":
    test_benchmark_peak_selection()
