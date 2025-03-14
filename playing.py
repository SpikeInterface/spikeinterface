from spikeinterface.generation import generate_drifting_recording
from spikeinterface.preprocessing.motion import correct_motion
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.generation.session_displacement_generator import generate_session_displacement_recordings
from spikeinterface.generation import generate_ground_truth_recording
from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_localization import localize_peaks


recordings_list, _ = generate_session_displacement_recordings(
    num_units=5,
    recording_durations=[1, 1],
    recording_shifts=((0, 0), (0, 250)),
    # TODO: can see how well this is recaptured by comparing the displacements to the known displacement + gradient
    non_rigid_gradient=None,  # 0.1, # 0.1,
    seed=55,  # 52
    generate_sorting_kwargs=dict(firing_rates=(100, 250), refractory_period_ms=4.0),
    generate_unit_locations_kwargs=dict(
        margin_um=0.0,
        # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
        minimum_z=0.0,
        maximum_z=2.0,
        minimum_distance=18.0,
        max_iteration=100,
        distance_strict=False,
    ),
    generate_noise_kwargs=dict(noise_levels=(0.0, 1.0), spatial_decay=1.0),
)
rec = recordings_list[1]

detect_kwargs = {
    "method": "locally_exclusive",
    "peak_sign": "neg",
    "detect_threshold": 25,
    "exclude_sweep_ms": 0.1,
    "radius_um": 75,
    "noise_levels": None,
    "random_chunk_kwargs": {},
}
localize_peaks_kwargs = {"method": "grid_convolution"}

# noise_levels = get_noise_levels(rec, return_scaled=False)
rec_0 = recordings_list[0]
rec_1 = recordings_list[1]

peaks_before_0 = detect_peaks(rec_0, **detect_kwargs)  # noise_levels=noise_levels,
peaks_before_1 = detect_peaks(rec_1, **detect_kwargs)

proc_rec_0, motion_info_0 = correct_motion(rec_0, preset="rigid_fast", detect_kwargs=detect_kwargs, localize_peaks_kwargs=localize_peaks_kwargs, output_motion_info=True)
proc_rec_1, motion_info_1 = correct_motion(rec_1, preset="rigid_fast", detect_kwargs=detect_kwargs, localize_peaks_kwargs=localize_peaks_kwargs, output_motion_info=True)

peaks_after_0 = detect_peaks(proc_rec_0, **detect_kwargs)  #  noise_levels=noise_levels
peaks_after_1 = detect_peaks(proc_rec_1, **detect_kwargs)


import spikeinterface.full as si
import matplotlib.pyplot as plt

# TODO: need to test multi-shank
plot = si.plot_traces(rec_1, order_channel_by_depth=True) # , time_range=(0, 0.1))
x = peaks_before_1["sample_index"] * (1/ rec_1.get_sampling_frequency())
y = rec_1.get_channel_locations()[peaks_before_1["channel_index"], 1]
plot.ax.scatter(x, y, color="r", s=2)
plt.show()

plot = si.plot_traces(proc_rec_1, order_channel_by_depth=True)
x = peaks_after_1["sample_index"] * (1/ proc_rec_1.get_sampling_frequency())
y = rec_1.get_channel_locations()[peaks_after_1["channel_index"], 1]
plot.ax.scatter(x, y, color="r", s=2)
plt.show()

breakpoint()
