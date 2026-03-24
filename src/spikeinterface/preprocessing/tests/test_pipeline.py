from spikeinterface.generation import generate_recording, generate_ground_truth_recording
from spikeinterface.preprocessing import (
    apply_preprocessing_pipeline,
    preprocessor_dict,
    bandpass_filter,
    common_reference,
    whiten,
    detect_and_remove_bad_channels,
    bandpass_filter,
    common_reference,
)
from spikeinterface.preprocessing.pipeline import (
    pp_names_to_functions,
    get_preprocessing_dict_from_file,
    get_preprocessing_dict_from_analyzer,
)
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core import create_sorting_analyzer


def test_pipeline_equiv_to_step():
    """
    For each preprocessing step, we create a preprocessed recording using 1) the apply_pipline function
    2) the preprocessor classes. We test that the two outputs are equal when the input is either
    a recording or a dict of recording.
    """

    single_group_rec = generate_recording(durations=[1])

    rec_groups = generate_recording(durations=[1])
    rec_groups.set_property(key="group", values=[0, 1])

    # Set some properties that some of the preprocessing steps rely on
    single_group_rec.set_property("gain_to_physical_unit", [2.0, 1.5])
    single_group_rec.set_property("offset_to_physical_unit", [0.0, 1.0])
    single_group_rec.set_property("gain_to_uV", [2.0, 1.5])
    single_group_rec.set_property("offset_to_uV", [0.0, 1.0])

    rec_groups.set_property("gain_to_physical_unit", [2.0, 1.5])
    rec_groups.set_property("offset_to_physical_unit", [0.0, 1.0])
    rec_groups.set_property("gain_to_uV", [2.0, 1.5])
    rec_groups.set_property("offset_to_uV", [0.0, 1.0])

    for rec in [single_group_rec, rec_groups.split_by("group")]:
        for _, pp_wrapper in preprocessor_dict.items():

            pp_name = pp_wrapper.__name__
            pp_class = pp_names_to_functions[pp_name]

            # each of these methods require a more bespoke recording, or a deep learning model
            if pp_name in [
                "phase_shift",
                "deepinterpolate",
                "highpass_spatial_filter",
                "unsigned_to_signed",
                "interpolate_bad_channels",
            ]:
                continue

            pp_dict = {pp_name: {}}

            if pp_name in ["normalize_by_quantile", "center", "zscore", "whiten"]:
                pp_dict[pp_name] = {"seed": 1205}
                pp_rec_from_class = pp_class(rec, seed=1205)
            elif pp_name == "blank_saturation":
                pp_dict[pp_name] = {"abs_threshold": 1.0}
                pp_rec_from_class = pp_class(rec, abs_threshold=1.0)
            elif pp_name == "silence_periods":
                pp_dict[pp_name] = {"list_periods": [(0, 0.5)]}
                pp_rec_from_class = pp_class(rec, list_periods=[(0, 0.5)])
            elif pp_name == "remove_artifacts":
                pp_dict[pp_name] = {"list_triggers": [0.5]}
                pp_rec_from_class = pp_class(rec, list_triggers=[0.5])
            elif pp_name == "zero_channel_pad":
                pp_dict[pp_name] = {"num_channels": 2}
                pp_rec_from_class = pp_class(rec, num_channels=2)
            elif pp_name == "resample":
                pp_dict[pp_name] = {"resample_rate": 10_000}
                pp_rec_from_class = pp_class(rec, resample_rate=10_000)
            elif pp_name == "decimate":
                pp_dict[pp_name] = {"decimation_factor": 2}
                pp_rec_from_class = pp_class(rec, decimation_factor=2)
            elif pp_name == "filter":
                pp_dict[pp_name] = {"margin_ms": 5.0}
                pp_rec_from_class = pp_class(rec, margin_ms=5.0)
            else:
                pp_rec_from_class = pp_class(rec)

            pp_rec_from_pipeline = apply_preprocessing_pipeline(rec, pp_dict)

            if isinstance(pp_rec_from_pipeline, dict):
                check_recordings_equal(pp_rec_from_pipeline[0], pp_rec_from_class[0])
                check_recordings_equal(pp_rec_from_pipeline[1], pp_rec_from_class[1])
            else:
                check_recordings_equal(pp_rec_from_pipeline, pp_rec_from_class)


def test_three_preprocessing_steps():
    """
    Apply three preprocessing steps using the `apply_pipline` function and doing
    so directly, then check the results are identical.
    """

    rec = generate_recording(durations=[1])

    pipeline_dict = {
        "common_reference": {},
        "bandpass_filter": {},
        "whiten": {"seed": 1205},
    }

    pp_rec_from_pipeline = apply_preprocessing_pipeline(rec, pipeline_dict)
    pp_rec_from_functions = whiten(bandpass_filter(common_reference(rec)), seed=1205)

    check_recordings_equal(pp_rec_from_pipeline, pp_rec_from_functions)

    rec_groups = generate_recording(durations=[1])
    rec_groups.set_property(key="group", values=[0, 1])
    dict_of_recs = rec_groups.split_by("group")

    pp_dict_of_recs_from_pipeline = apply_preprocessing_pipeline(dict_of_recs, pipeline_dict)
    pp_dict_of_recs_from_functions = whiten(bandpass_filter(common_reference(dict_of_recs)), seed=1205)

    check_recordings_equal(pp_dict_of_recs_from_pipeline[0], pp_dict_of_recs_from_functions[0])
    check_recordings_equal(pp_dict_of_recs_from_pipeline[1], pp_dict_of_recs_from_functions[1])


def test_kwargs_are_propagated():
    """
    Apply a preprocessing step, `bandpass_filter`, with default kwargs and non-default
    kwargs and check that arguments are propagated as expected.
    """

    rec = generate_recording(durations=[1])
    pipeline_dict = {"bandpass_filter": {}}

    bp_rec_default = apply_preprocessing_pipeline(rec, pipeline_dict)

    kwargs = bp_rec_default._kwargs
    assert kwargs["freq_min"] == 300.0

    pipeline_dict_non_default = {"bandpass_filter": {"freq_min": 500.0}}

    bp_rec_non_default = apply_preprocessing_pipeline(rec, pipeline_dict_non_default)
    non_default_kwargs = bp_rec_non_default._kwargs

    assert non_default_kwargs["freq_min"] == 500.0


def test_loading_provenance(create_cache_folder):
    """
    Makes a preprocessed recording using a Pipeline and saves it. Then reloads the preprocessed
    recording using `get_preprocessing_dict_from_file`, either ignoring or applying the
    precomputed kwargs. These reloaded recordings should be the same as the original preprocessed
    recording.
    """

    cache_folder = create_cache_folder / "preprocessed_rec_for_pipeline"

    rec, _ = generate_ground_truth_recording(seed=0, num_channels=6)
    pp_rec = detect_and_remove_bad_channels(
        bandpass_filter(common_reference(rec, operator="average")),
        noisy_channel_threshold=0.3,
        # this seed is for detect_bad_channels_kwargs this ensure the same random_chunk_kwargs
        # when several run
        seed=2205,
    )
    pp_rec.save_to_folder(folder=cache_folder)

    loaded_pp_dict = get_preprocessing_dict_from_file(cache_folder / "provenance.pkl")

    pipeline_rec_applying_precomputed_kwargs = apply_preprocessing_pipeline(
        rec,
        loaded_pp_dict,
        apply_precomputed_kwargs=True,
    )
    pipeline_rec_ignoring_precomputed_kwargs = apply_preprocessing_pipeline(
        rec, loaded_pp_dict, apply_precomputed_kwargs=False
    )

    check_recordings_equal(pipeline_rec_applying_precomputed_kwargs, pp_rec)
    check_recordings_equal(pipeline_rec_ignoring_precomputed_kwargs, pp_rec)


def test_loading_from_analyzer(create_cache_folder):
    """
    Tests the `get_preprocessing_dict_from_analyzer` function, which constructs a preprocessing pipeline
    dict from a saved sorting analyzer (either binary folder or zarr). This test creates a preprocessed recording,
    uses this to create a sorting analyzer and saves binary and zarr versions of the analyzer. Then we generate
    the preprocessing dict from the analyzer, and apply it to the original recording to check that it's the same
    as the preprocessed recording made earlier.
    """

    cache_folder = create_cache_folder
    recording, sorting = generate_ground_truth_recording()

    preprocessing_dict = {"common_reference": {}, "highpass_filter": {"freq_min": 301.0}}
    pp_recording = apply_preprocessing_pipeline(recording, preprocessing_dict)

    analyzer_binary_folder = cache_folder / "binary_format"
    _ = create_sorting_analyzer(
        sorting=sorting, recording=pp_recording, format="binary_folder", folder=analyzer_binary_folder
    )
    pp_dict_from_binary = get_preprocessing_dict_from_analyzer(analyzer_binary_folder)
    pp_recording_from_binary = apply_preprocessing_pipeline(recording, pp_dict_from_binary)
    check_recordings_equal(pp_recording, pp_recording_from_binary)

    analyzer_zarr_folder = cache_folder / "zarr_format.zarr"
    _ = create_sorting_analyzer(sorting=sorting, recording=pp_recording, format="zarr", folder=analyzer_zarr_folder)
    pp_dict_from_zarr = get_preprocessing_dict_from_analyzer(analyzer_zarr_folder)
    pp_recording_from_zarr = apply_preprocessing_pipeline(recording, pp_dict_from_zarr)
    check_recordings_equal(pp_recording, pp_recording_from_zarr)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    tmp_folder = Path(tempfile.mkdtemp())
    test_loading_provenance(tmp_folder)
