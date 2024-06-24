# # Benchmark spike sorting with hybrid recordings
#
# This example shows how to use the SpikeInterface hybrid recordings framework to benchmark spike sorting results.
#
# Hybrid recordings are built from existing recordings by injecting units with known spiking activity.
# The template (aka average waveforms) of the injected units can be from previous spike sorted data.
# In this example, we will be using an open database of templates that we have constructed from the International Brain Laboratory - Brain Wide Map (available on [DANDI](https://dandiarchive.org/dandiset/000409?search=IBL&page=2&sortOption=0&sortDir=-1&showDrafts=true&showEmpty=false&pos=9)).
#
# Importantly, recordings from long-shank probes, such as Neuropixels, usually experience drifts. Such drifts have to be taken into account in order to smoothly inject spikes into the recording.

# +
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import spikeinterface.generation as sgen
import spikeinterface.widgets as sw

from spikeinterface.sortingcomponents.motion_estimation import estimate_motion

from pathlib import Path
# -

# %matplotlib inline

# To make this notebook self-contained, we will simulate a drifting recording similar to the one acquired by Nick Steinmetz and available [here](https://doi.org/10.6084/m9.figshare.14024495.v1), where an triangular motion was imposed to the recording by moving the probe up and down with a micro-manipulator.

# +
zigzag_rigid = [
    {
        'drift_mode': 'zigzag',
        'non_rigid_gradient': None,
        't_start_drift': 30.0,
        't_end_drift': 210,
        'period_s': 60
    }
]

# this generates a "static" and "drifting" recording version
static_recording, drifting_recording, sorting = sgen.generate_drifting_recording(
    probe_name="Neuropixel-128", #,'Neuropixel-384',
    seed=23,
    duration=240,
    num_units=100,
    generate_displacement_vector_kwargs={'motion_list' : zigzag_rigid}
)
# -

# To visualize the

_, motion_info = spre.correct_motion(
    drifting_recording,
    preset="nonrigid_fast_and_accurate",
    n_jobs=4,
    progress_bar=True,
    output_motion_info=True
)
