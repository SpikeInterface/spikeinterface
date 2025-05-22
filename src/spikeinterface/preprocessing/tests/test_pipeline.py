from spikeinterface.generation import generate_recording
from spikeinterface.preprocessing import apply_pipeline, preprocessor_dict, bandpass_filter, common_reference, whiten
from spikeinterface.preprocessing.pipeline import pp_names_to_functions
from spikeinterface.core.testing import check_recordings_equal


def test_pipeline_equiv_to_step():
    """
    For each preprocessing step, we create a preprocessed recording using 1) the apply_pipline function
    2) the preprocessor classes. We test that the two outputs are equal when the input is either
    a recording or a dict of recording.
    """

    singl_group_rec = generate_recording(durations=[1])

    rec_groups = generate_recording(durations=[1])
    rec_groups.set_property(key="group", values=[0, 1])

    for rec in (singl_group_rec, rec_groups.split_by("group")):
        for _, pp_wrapper in preprocessor_dict.items():

            pp_name = pp_wrapper.__name__
            pp_class = pp_names_to_functions[pp_name]

            # each of these methods require a more bespoke recording, or a deep learning model
            if pp_name in [
                "phase_shift",
                "deepinterpolate",
                "highpass_spatial_filter",
                "interpolate_bad_channels",
                "unsigned_to_signed",
                "filter",
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
            else:
                pp_rec_from_class = pp_class(rec)

            pp_rec_from_pipeline = apply_pipeline(rec, pp_dict)

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

    pp_rec_from_pipeline = apply_pipeline(rec, pipeline_dict)
    pp_rec_from_functions = whiten(bandpass_filter(common_reference(rec)), seed=1205)

    check_recordings_equal(pp_rec_from_pipeline, pp_rec_from_functions)

    rec_groups = generate_recording(durations=[1])
    rec_groups.set_property(key="group", values=[0, 1])
    dict_of_recs = rec_groups.split_by("group")

    pp_dict_of_recs_from_pipeline = apply_pipeline(dict_of_recs, pipeline_dict)
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

    bp_rec_default = apply_pipeline(rec, pipeline_dict)

    kwargs = bp_rec_default._kwargs
    assert kwargs["freq_min"] == 300.0

    pipeline_dict_non_default = {"bandpass_filter": {"freq_min": 500.0}}

    bp_rec_non_default = apply_pipeline(rec, pipeline_dict_non_default)
    non_default_kwargs = bp_rec_non_default._kwargs

    assert non_default_kwargs["freq_min"] == 500.0
