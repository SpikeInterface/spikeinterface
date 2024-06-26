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

import numpy as np
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
    probe_name="Neuropixel-384", #,'Neuropixel-384',
    seed=23,
    duration=240,
    num_units=100,
    generate_displacement_vector_kwargs={'motion_list' : zigzag_rigid}
)
# -

# To visualize the drift, we can estimate the motion and plot it:

_, motion_info = spre.correct_motion(
    drifting_recording,
    preset="nonrigid_fast_and_accurate",
    n_jobs=4,
    progress_bar=True,
    output_motion_info=True
)

# +
#
# sw.plot_drift_map(...)
#
# -

# ## Retrieve templates from database

templates_info = sgen.fetch_templates_database_info()

templates_info.head()

available_brain_areas = np.unique(templates_info.brain_area)
print(f"Available brain areas: {available_brain_areas}")

# let's perform a query: templates from brain region VISp5 and at the "top" of the probe
target_area = "VISp5"
minimum_depth = 1500
templates_selected_info = templates_info.query(
    f"brain_area == '{target_area}' and depth_along_probe > {minimum_depth}"
)
display(templates_selected_info)

# We can now retrieve the selected templates as a `Templates` object
#

templates_selected = sgen.query_templates_from_database(templates_selected_info, verbose=True)
print(templates_selected)

# While we selected templates from a target aread and at certain depths, we can see that the template amplitudes are quite large. This will make spike sorting easy... we can further manipulate the `Templates` by rescaling, relocating, or further selections with the `sgen.scale_template_to_range`, `sgen.relocate_templates`, and `sgen.select_templates` functions.
#
# In our case, let's rescale the amplitudes between 30 and 50 $\mu$V.

templates_scaled = sgen.scale_template_to_range(templates=templates_selected, min_amplitude=30, max_amplitude=50)

# Let's plot the selected templates:

w.figure.su

sparsity_plot = si.compute_sparsity(templates_selected)
w = sw.plot_unit_templates(templates_scaled, sparsity=sparsity_plot, ncols=4)
w.figure.subplots_adjust(wspace=0.5, hspace=0.7)

# ## Constructing hybrid recordings

drifting_recording = spre.depth_order(drifting_recording)

drifting_recording.get_channel_locations()[:10]

templates_scaled.get_channel_locations()[:10]

recording_hybrid, sorting_hybrid = sgen.generate_hybrid_recording(
    recording=drifting_recording,
    templates=templates_scaled,
    motion=motion_info["motion"],
    seed=2308
)

recording_hybrid

# +
# show spike locations on top of original rastermap
