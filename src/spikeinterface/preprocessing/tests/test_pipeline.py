from spikeinterface.generation import generate_recording
from spikeinterface.preprocessing import apply_pipeline, preprocessor_dict
from spikeinterface.preprocessing.pipeline import pp_names_to_classes
from spikeinterface.core.testing import check_recordings_equal


def test_pipeline_equiv_to_step():
    """
    For each preprocessing step, we create a preprocessed recording using 1) the apply_pipline function
    2) the preprocessor classes. We test that the two outputs are equal.
    """

    rec = generate_recording(durations=[10])

    for _, pp_wrapper in preprocessor_dict.items():

        pp_name = pp_wrapper.__name__
        pp_class = pp_names_to_classes[pp_name]

        # each of these methods require a more bespoke recording, or a deep learning model
        if pp_name in [
            "phase_shift",
            "deepinterpolate",
            "highpass_spatial_filter",
            "interpolate_bad_channels",
            "unsigned_to_signed",
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
            pp_dict[pp_name] = {"list_periods": [(0, 1)]}
            pp_rec_from_class = pp_class(rec, list_periods=[(0, 1)])
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

        check_recordings_equal(pp_rec_from_pipeline, pp_rec_from_class)
