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
import matplotlib.pyplot as plt
from pathlib import Path

# -

# %matplotlib inline

si.set_global_job_kwargs(n_jobs=4)

# To make this notebook self-contained, we will simulate a drifting recording similar to the one acquired by Nick Steinmetz and available [here](https://doi.org/10.6084/m9.figshare.14024495.v1), where an triangular motion was imposed to the recording by moving the probe up and down with a micro-manipulator.

# +
# sgen.generate_displacement_vector?

# +
generate_displacement_vector_kwargs = {
    "motion_list": [
        {
            "drift_mode": "zigzag",
            "non_rigid_gradient": None,
            "t_start_drift": 30.0,
            "t_end_drift": 210,
            "period_s": 60,
        }
    ],
    "drift_start_um": [0, 100],
    "drift_stop_um": [0, -100],
}

# this generates a "static" and "drifting" recording version
static_recording, drifting_recording, sorting = sgen.generate_drifting_recording(
    probe_name="Neuropixel-384",  # ,'Neuropixel-384',
    seed=23,
    duration=240,
    num_units=100,
    generate_displacement_vector_kwargs=generate_displacement_vector_kwargs,
)

# we sort the channels by depth, to match the  hybrid templates
drifting_recording = spre.depth_order(drifting_recording)
# -

# To visualize the drift, we can estimate the motion and plot it:

_, motion_info = spre.correct_motion(
    drifting_recording, preset="nonrigid_fast_and_accurate", n_jobs=4, progress_bar=True, output_motion_info=True
)

sw.plot_motion_info(motion_info, recording=drifting_recording)


def plot_drift_map(
    peaks=None,
    peak_locations=None,
    recording=None,
    analyzer=None,
    direction="y",
    sampling_frequency=None,
    segment_index=0,
    depth_lim=None,
    color_amplitude=True,
    scatter_decimate=None,
    color="k",
    cmap="inferno",
    clim=None,
    alpha=1,
    ax=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if ax is None:
        fig, ax = plt.subplots()

    assert peaks is not None or analyzer is not None
    if peaks is not None:
        assert peak_locations is not None
        if recording is None:
            assert sampling_frequency is not None
        else:
            sampling_frequency = recording.sampling_frequency
        peak_amplitudes = peaks["amplitude"]
    if analyzer is not None:
        if analyzer.has_recording():
            recording = analyzer.recording
        else:
            recording = None
            sampling_frequency = analyzer.sampling_frequency
        peaks = analyzer.sorting.to_spike_vector()
        assert analyzer.has_extension("spike_locations")
        peak_locations = analyzer.get_extension("spike_locations").get_data()
        if analyzer.has_extension("spike_amplitudes"):
            peak_amplitudes = analyzer.get_extension("spike_amplitudes").get_data()
        else:
            peak_amplitudes = None
    times = recording.get_times(segment_index=segment_index) if recording is not None else None

    if times is None:
        x = peaks["sample_index"] / sampling_frequency
    else:
        # use real times and adjust temporal bins with t_start
        x = times[peaks["sample_index"]]

    y = peak_locations[direction]
    if scatter_decimate is not None:
        x = x[::scatter_decimate]
        y = y[::scatter_decimate]
        y2 = y2[::scatter_decimate]

    if color_amplitude:
        assert peak_amplitudes is not None, "To color by amplitudes the 'spike_amplitude' extension is needed"
        amps = peak_amplitudes
        amps_abs = np.abs(amps)
        q_95 = np.quantile(amps_abs, 0.95)
        if scatter_decimate is not None:
            amps = amps[::scatter_decimate]
            amps_abs = amps_abs[::scatter_decimate]
        cmap = plt.colormaps[cmap]
        if clim is None:
            amps = amps_abs
            amps /= q_95
            c = cmap(amps)
        else:
            norm_function = Normalize(vmin=clim[0], vmax=clim[1], clip=True)
            c = cmap(norm_function(amps))
        color_kwargs = dict(
            color=None,
            c=c,
            alpha=alpha,
        )
    else:
        color_kwargs = dict(color=color, c=None, alpha=alpha)

    ax.scatter(x, y, s=1, **color_kwargs)
    if depth_lim is not None:
        ax.set_ylim(*depth_lim)
    ax.set_title("Peak depth")
    ax.set_xlabel("Times [s]")
    ax.set_ylabel("Depth [$\\mu$m]")
    return ax


ax = plot_drift_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=drifting_recording,
    cmap="Greys_r",
)

# +
#
# sw.plot_drift_map(...)
#
# -

# ## Retrieve templates from database

templates_info = sgen.fetch_templates_database_info()

print(len(templates_info))

templates_info.head()

available_brain_areas = np.unique(templates_info.brain_area)
print(f"Available brain areas: {available_brain_areas}")

# let's perform a query: templates from brain region VISp5 and at the "top" of the probe
target_area = ["VISa5", "VISa6a", "VISp5", "VISp6a", "VISrl6b"]
minimum_depth = 1000
templates_selected_info = templates_info.query(f"brain_area in {target_area} and depth_along_probe > {minimum_depth}")
display(templates_selected_info)

# We can now retrieve the selected templates as a `Templates` object
#

templates_selected = sgen.query_templates_from_database(templates_selected_info, verbose=True)
print(templates_selected)

# While we selected templates from a target aread and at certain depths, we can see that the template amplitudes are quite large. This will make spike sorting easy... we can further manipulate the `Templates` by rescaling, relocating, or further selections with the `sgen.scale_template_to_range`, `sgen.relocate_templates`, and `sgen.select_templates` functions.
#
# In our case, let's rescale the amplitudes between 30 and 50 $\mu$V.

templates_scaled = sgen.scale_template_to_range(templates=templates_selected, min_amplitude=30, max_amplitude=50)

# +
# sgen.relocate_templates?
# -

templates_relocated = sgen.relocate_templates(templates=templates_scaled, min_displacement=200, max_displacement=2000)

# Let's plot the selected templates:

sparsity_plot = si.compute_sparsity(templates_relocated)
w = sw.plot_unit_templates(templates_relocated, sparsity=sparsity_plot, ncols=4)
w.figure.subplots_adjust(wspace=0.5, hspace=0.7)

# ## Constructing hybrid recordings

recording_hybrid_no_drift, sorting_hybrid = sgen.generate_hybrid_recording(
    recording=drifting_recording, templates=templates_relocated, seed=2308
)

recording_hybrid_no_drift

recording_hybrid, sorting_hybrid = sgen.generate_hybrid_recording(
    recording=drifting_recording,
    templates=templates_relocated,
    motion=motion_info["motion"],
    sorting=sorting_hybrid,
    seed=2308,
)

recording_hybrid

# show spike locations on top of original rastermap
# construct hybrid analyzer for spike locations
analyzer_hybrid = si.create_sorting_analyzer(sorting_hybrid, recording_hybrid)
analyzer_hybrid.compute(["random_spikes", "templates"])
analyzer_hybrid.compute("spike_locations", method="grid_convolution")

# construct hybrid analyzer for spike locations
analyzer_hybrid_no_drift = si.create_sorting_analyzer(sorting_hybrid, recording_hybrid_no_drift)
analyzer_hybrid_no_drift.compute(["random_spikes", "templates"])
analyzer_hybrid_no_drift.compute("spike_locations", method="grid_convolution")

fig, axs = plt.subplots(ncols=2, figsize=(10, 7))
_ = plot_drift_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=drifting_recording,
    cmap="Greys_r",
    ax=axs[0],
)
_ = plot_drift_map(analyzer=analyzer_hybrid_no_drift, color_amplitude=False, color="r", ax=axs[0])
_ = plot_drift_map(
    peaks=motion_info["peaks"],
    peak_locations=motion_info["peak_locations"],
    recording=drifting_recording,
    cmap="Greys_r",
    ax=axs[1],
)
_ = plot_drift_map(analyzer=analyzer_hybrid, color_amplitude=False, color="b", ax=axs[1])
axs[0].set_title("Ignoring drift")
axs[1].set_title("Accounting for drift")

# ## Ground-truth study
#
# In this section ...

# +
datasets = {
    "hybrid": (recording_hybrid, sorting_hybrid),
}

cases = {
    ("kilosort2.5", "hybrid"): {
        "label": "KS2.5",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "kilosort2_5",
        },
    },
    ("kilosort3", "hybrid"): {
        "label": "KS3",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "kilosort3",
        },
    },
    ("kilosort4", "hybrid"): {
        "label": "KS4",
        "dataset": "hybrid",
        "run_sorter_params": {"sorter_name": "kilosort4", "nblocks": 5},
    },
    ("sc2", "hybrid"): {
        "label": "spykingcircus2",
        "dataset": "hybrid",
        "run_sorter_params": {
            "sorter_name": "spykingcircus2",
        },
    },
}
# -

study_folder = workdir / "gt_study"
if (workdir / "gt_study").is_dir():
    gtstudy = sc.GroundTruthStudy(study_folder)
else:
    gtstudy = sc.GroundTruthStudy.create(study_folder=study_folder, datasets=datasets, cases=cases)

gtstudy.run_sorters(verbose=True, keep=True)

gtstudy.run_comparisons(exhaustive_gt=False)

w_run_times = sw.plot_study_run_times(gtstudy)
w_perf = sw.plot_study_performances(gtstudy, figsize=(12, 7))
w_perf.axes[0, 0].legend(loc=4)
