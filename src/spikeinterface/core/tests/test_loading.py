import pytest

import numpy as np
from spikeinterface import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
    load,
    SortingAnalyzer,
    Templates,
    aggregate_channels,
)
from spikeinterface.core.motion import Motion
from spikeinterface.core.generate import generate_unit_locations, generate_templates
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


@pytest.fixture()
def generate_templates_object():
    from probeinterface import generate_linear_probe

    seed = 0

    num_chans = 12
    num_units = 10
    margin_um = 15.0
    probe = generate_linear_probe(num_chans)
    channel_locations = probe.contact_positions[:, :2]
    unit_locations = generate_unit_locations(num_units, channel_locations, margin_um=margin_um, seed=seed)

    sampling_frequency = 30000.0
    ms_before = 1.0
    ms_after = 3.0

    # standard case
    templates_arr = generate_templates(
        channel_locations,
        unit_locations,
        sampling_frequency,
        ms_before,
        ms_after,
        upsample_factor=None,
        seed=42,
        dtype="float32",
    )
    templates = Templates(
        templates_array=templates_arr,
        sampling_frequency=sampling_frequency,
        nbefore=int(ms_before * sampling_frequency / 1000),
        probe=probe,
    )
    return templates


@pytest.fixture()
def generate_motion_object():
    num_spatial_bins = 10
    num_temporal_bins = 100
    temporal_bins_s = np.arange(num_temporal_bins)
    spatial_bins_um = np.arange(num_spatial_bins)
    displacements = np.random.randn(num_temporal_bins, num_spatial_bins)
    motion = Motion(displacements, temporal_bins_s=temporal_bins_s, spatial_bins_um=spatial_bins_um)
    return motion


@pytest.mark.parametrize("output_format", ["binary", "zarr"])
def test_load_binary_recording(generate_recording_sorting, tmp_path, output_format):
    rec, _ = generate_recording_sorting
    _ = rec.save(folder=tmp_path / "test_recording", format=output_format, overwrite=True)

    if output_format == "zarr":
        rec_loaded = load(tmp_path / "test_recording.zarr")
    else:
        rec_loaded = load(tmp_path / "test_recording")

    check_recordings_equal(rec, rec_loaded)


@pytest.mark.parametrize("output_format", ["numpy_folder", "zarr"])
def test_load_binary_sorting(generate_recording_sorting, tmp_path, output_format):
    _, sort = generate_recording_sorting
    _ = sort.save(folder=tmp_path / "test_sorting", format=output_format, overwrite=True)

    if output_format == "zarr":
        sort_loaded = load(tmp_path / "test_sorting.zarr")
    else:
        sort_loaded = load(tmp_path / "test_sorting")

    check_sortings_equal(sort, sort_loaded)
    del sort


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


def test_load_templates(tmp_path, generate_templates_object):
    templates = generate_templates_object
    templates_dict = templates.to_dict()
    templates_loaded = load(templates_dict)
    assert templates == templates_loaded

    zarr_path = tmp_path / "templates.zarr"
    templates.to_zarr(str(zarr_path))
    # Load from the Zarr archive
    templates_loaded = load(str(zarr_path))

    assert templates == templates_loaded


def test_load_motion(tmp_path, generate_motion_object):
    motion = generate_motion_object

    motion_dict = motion.to_dict()
    motion_loaded = load(motion_dict)
    assert motion == motion_loaded

    motion.save(tmp_path / "motion")
    motion_loaded = load(tmp_path / "motion")

    assert motion == motion_loaded


def test_load_aggregate_recording_from_json(generate_recording_sorting, tmp_path):
    """
    Save, then load an aggregated recording using its provenance.json file.
    """

    recording, _ = generate_recording_sorting

    recording.set_property("group", [0, 0, 1, 1])
    list_of_recs = list(recording.split_by("group").values())
    aggregated_rec = aggregate_channels(list_of_recs)

    recording_path = tmp_path / "aggregated_recording"
    aggregated_rec.save_to_folder(folder=recording_path)
    loaded_rec = load(recording_path / "provenance.json", base_folder=recording_path)

    assert np.all(loaded_rec.get_property("group") == recording.get_property("group"))


@pytest.mark.streaming_extractors
@pytest.mark.skipif(not HAVE_S3, reason="s3fs not installed")
def test_remote_recording():
    s3_path = "s3://spikeinterface-sorting-analyzer-test/recording_for_analyzer_short.zarr/"
    rec = load(s3_path)
    assert isinstance(rec, ZarrRecordingExtractor)


@pytest.mark.streaming_extractors
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
