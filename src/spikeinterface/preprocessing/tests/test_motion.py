import shutil
import pytest

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import (
    correct_motion,
    load_motion_info,
    save_motion_info,
    get_motion_parameters_preset,
    compute_motion,
)
from spikeinterface.preprocessing.motion import _get_default_motion_params


def test_estimate_and_correct_motion(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(durations=[30.0], num_channels=12)

    folder = cache_folder / "estimate_and_correct_motion"
    if folder.exists():
        shutil.rmtree(folder)

    rec_corrected = correct_motion(rec, folder=folder, estimate_motion_kwargs={"win_step_um": 50, "win_scale_um": 100})

    # test reloading motion info
    motion_info = load_motion_info(folder)

    # test saving motion info
    save_folder = folder / "motion_info"
    save_motion_info(motion_info=motion_info, folder=save_folder)
    motion_info_loaded = load_motion_info(save_folder)
    assert motion_info_loaded["motion"] == motion_info["motion"]


def test_estimate_and_correct_motion_int():

    rec = generate_recording(durations=[30.0], num_channels=12).astype(int)
    rec_corrected = correct_motion(rec, estimate_motion_kwargs={"win_step_um": 50, "win_scale_um": 100})
    assert rec_corrected.get_dtype().kind == "f"


def test_get_motion_parameters_preset():
    from pprint import pprint

    p = _get_default_motion_params()
    # pprint(p)

    params = get_motion_parameters_preset("nonrigid_accurate")
    params = get_motion_parameters_preset("dredge")
    params = get_motion_parameters_preset("rigid_fast")
    pprint(params)


def test_compute_motion_fails(create_cache_folder):
    """
    If motion estimation fails, `compute_motion` should still return a `motion_info` dict with all information except
    the motion object. This tests whether this does happen, then checks that saving and loading still works when motion
    is `None`.
    """
    # this recording has too few channels for the motion correction to work
    rec = generate_recording(durations=[5], num_channels=1)
    motion, motion_info = compute_motion(rec, raise_error=False, output_motion_info=True)
    print(motion)

    assert motion is None
    assert motion_info["motion"] is None
    assert motion_info["peaks"] is not None
    assert motion_info["parameters"] is not None

    # save and load the motion. We expect a warning.
    cache_folder = create_cache_folder
    folder = cache_folder / "motion_info_with_motion_none"
    save_motion_info(motion_info, folder=folder)

    with pytest.warns(UserWarning):
        loaded_motion_info = load_motion_info(folder)

    assert loaded_motion_info["motion"] is None


if __name__ == "__main__":
    # print(correct_motion.__doc__)
    # test_estimate_and_correct_motion()
    # test_get_motion_parameters_preset()
    from pathlib import Path

    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "preprocessing"
    test_compute_motion_fails(cache_folder)
