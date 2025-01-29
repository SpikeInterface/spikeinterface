import pytest

from spikeinterface import generate_ground_truth_recording, create_sorting_analyzer, load, SortingAnalyzer
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.core.zarrextractors import ZarrRecordingExtractor

try:
    import s3fs

    HAVE_S3 = True
except ImportError:
    HAVE_S3 = False

computed_extensions = ["waveforms", "templates", "noise_levels", "random_spikes"]


@pytest.fixture()
def generate_recording_sorting(tmp_path):
    rec, sort = generate_ground_truth_recording(num_channels=4, durations=[2, 3], seed=0)
    rec = rec.save(folder=tmp_path / "recording", overwrite=True)
    sort = sort.save(folder=tmp_path / "sorting", overwrite=True)
    return rec, sort


@pytest.fixture()
def generate_sorting_analyzer(generate_recording_sorting):
    rec, sort = generate_recording_sorting
    analyzer = create_sorting_analyzer(sort, recording=rec)
    analyzer.compute(computed_extensions)
    return analyzer


@pytest.mark.parametrize("output_format", ["binary", "zarr"])
def test_load_binary_recording(generate_recording_sorting, tmp_path, output_format):
    rec, _ = generate_recording_sorting
    _ = rec.save(folder=tmp_path / "recording", format=output_format, overwrite=True)

    if output_format == "zarr":
        rec_loaded = load(tmp_path / "recording.zarr")
    else:
        rec_loaded = load(tmp_path / "recording")

    check_recordings_equal(rec, rec_loaded)


@pytest.mark.parametrize("output_format", ["numpy_folder", "zarr"])
def test_load_binary_sorting(generate_recording_sorting, tmp_path, output_format):
    _, sort = generate_recording_sorting
    _ = sort.save(folder=tmp_path / "sorting", format=output_format, overwrite=True)

    if output_format == "zarr":
        sort_loaded = load(tmp_path / "sorting.zarr")
    else:
        sort_loaded = load(tmp_path / "sorting")

    check_sortings_equal(sort, sort_loaded)


@pytest.mark.parametrize("extension", [".pkl", ".json"])
def test_load_ext_extractors(generate_recording_sorting, tmp_path, extension):
    rec, sort = generate_recording_sorting

    _ = rec.dump(tmp_path / f"recording{extension}")
    _ = sort.dump(tmp_path / f"sorting{extension}")

    rec_loaded = load(tmp_path / f"recording{extension}")
    sort_loaded = load(tmp_path / f"sorting{extension}")

    check_recordings_equal(rec, rec_loaded, check_properties=False)
    check_sortings_equal(sort, sort_loaded, check_properties=False)

    # now with relative_to
    _ = rec.dump(tmp_path / f"recording{extension}", relative_to=tmp_path)
    _ = sort.dump(tmp_path / f"sorting{extension}", relative_to=tmp_path)

    rec_loaded = load(tmp_path / f"recording{extension}", base_folder=tmp_path)
    sort_loaded = load(tmp_path / f"sorting{extension}", base_folder=tmp_path)

    check_recordings_equal(rec, rec_loaded, check_properties=False)
    check_sortings_equal(sort, sort_loaded, check_properties=False)


@pytest.mark.parametrize("output_format", ["binary_folder", "zarr"])
def test_load_sorting_analyzer(generate_sorting_analyzer, tmp_path, output_format):
    analyzer = generate_sorting_analyzer
    _ = analyzer.save_as(folder=tmp_path / "analyzer", format=output_format)

    if output_format == "zarr":
        analyzer_loaded = load(tmp_path / "analyzer.zarr")
    else:
        analyzer_loaded = load(tmp_path / "analyzer")

    check_recordings_equal(analyzer.recording, analyzer_loaded.recording)
    check_sortings_equal(analyzer.sorting, analyzer_loaded.sorting)

    for ext in computed_extensions:
        assert ext in analyzer_loaded.extensions


@pytest.mark.skipif(not HAVE_S3, reason="s3fs not installed")
def test_remote_recording():
    s3_path = "s3://spikeinterface-sorting-analyzer-test/recording_for_analyzer_short.zarr/"
    rec = load(s3_path)
    assert isinstance(rec, ZarrRecordingExtractor)


@pytest.mark.skipif(not HAVE_S3, reason="s3fs not installed")
def test_remote_analyzer():
    s3_path = "s3://spikeinterface-sorting-analyzer-test/analyzer_remote_test.zarr/"
    analyzer = load(s3_path)
    assert isinstance(analyzer, SortingAnalyzer)
    for ext in [
        "noise_levels",
        "random_spikes",
        "templates",
        "correlograms",
        "spike_amplitudes",
        "template_metrics",
        "template_similarity",
        "unit_locations",
        "quality_metrics",
    ]:
        assert ext in analyzer.get_saved_extension_names()
