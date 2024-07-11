"""
Motion estimation
=================

SpikeInterface offers a very flexible framework to handle drift as a
preprocessing step. If you want to know more, please read the
:ref:`motion_correction` section of the documentation.

Here a short example with a simulated drifting recording.

"""

# %%
import matplotlib.pyplot as plt


from spikeinterface.generation import generate_drifting_recording
from spikeinterface.preprocessing import correct_motion
from spikeinterface.widgets import plot_motion, plot_motion_info, plot_probe_map

# %%
# First, let's simulate a drifting recording using the
# :code:`spikeinterface.generation module`.
#
# Here the simulated recording has a small zigzag motion along the 'y' axis of the probe.

static_recording, drifting_recording, sorting = generate_drifting_recording(
    num_units=200,
    duration=300.,
    probe_name='Neuropixel-128',
    generate_displacement_vector_kwargs=dict(
        displacement_sampling_frequency=5.0,
        drift_start_um=[0, 20],
        drift_stop_um=[0, -20],
        drift_step_um=1,
        motion_list=[
            dict(
                drift_mode="zigzag",
                non_rigid_gradient=None,
                t_start_drift=60.0,
                t_end_drift=None,
                period_s=200,
            ),
        ],
    ),
    seed=2205,
)

plot_probe_map(drifting_recording)

# %%
# Here we will use the high level function :code:`correct_motion()`
#
# Internally, this function is doing all steps of the motion detection:
#  1. **activity profile** : detect peaks and localize them along time and depth
#  2. **motion inference**: estimate the drift motion
#  3. **motion interpolation**: interpolate traces using the estimated motion
#
# All steps have an use several methods with many parameters. This is why we can use
# 'preset' which combine methods and related parameters.
#
# This function can take a while peak detection and localization is a slow process
# that need to go trought the entire traces

recording_corrected, motion, motion_info = correct_motion(
    drifting_recording, preset="nonrigid_fast_and_accurate",
    output_motion=True, output_motion_info=True,
    n_jobs=-1, progress_bar=True,
)

# %%
# The function return a recording 'corrected'
#
# A new recording is return, this recording will interpolate motion corrected traces
# when calling get_traces()

print(recording_corrected)

# %%
# Optionally the function also return the `Motion` object itself
#

print(motion)

# %%
# This motion can be plotted, in our case the motion has been estimated as non-rigid
# so we can use the use the `mode='map'` to check the motion across depth.
#

plot_motion(motion, mode='line')
plot_motion(motion, mode='map')


# %%
# The dict `motion_info` can be used for more plotting.
# Here we can appreciate of the two top axes the raster of peaks depth vs times before and
# after correction.

fig = plt.figure()
plot_motion_info(motion_info, drifting_recording, amplitude_cmap="inferno", color_amplitude=True, figure=fig)
fig.axes[0].set_ylim(520, 620)
plt.show()
# %%
