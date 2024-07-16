import shutil

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import correct_motion, load_motion_info, save_motion_info


def test_estimate_and_correct_motion(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(durations=[30.0], num_channels=12)
    print(rec)

    folder = cache_folder / "estimate_and_correct_motion"
    if folder.exists():
        shutil.rmtree(folder)

    rec_corrected = correct_motion(rec, folder=folder, estimate_motion_kwargs={"win_step_um": 50, "win_scale_um": 100})
    print(rec_corrected)

    # test reloading motion info
    motion_info = load_motion_info(folder)
    print(motion_info.keys())

    # test saving motion info
    save_folder = folder / "motion_info"
    save_motion_info(motion_info=motion_info, folder=save_folder)
    motion_info_loaded = load_motion_info(save_folder)
    assert motion_info_loaded["motion"] == motion_info["motion"]


if __name__ == "__main__":
    # print(correct_motion.__doc__)
    test_estimate_and_correct_motion()
