"""
SortingAnalyzer
===============

SpikeInterface provides an object to gather a Recording and a Sorting to perform various
analyses and visualizations of the sorting : :py:class:`~spikeinterface.core.SortingAnalyzer`.

This :py:class:`~spikeinterface.core.SortingAnalyzer` class:

  * is the first step for all post post processing, quality metrics, and visualization.
  * gathers a recording and a sorting
  * can be sparse or dense : (i.e. whether all channel are used for all units or not).
  * handle a list of "extensions"
  * "core extensions" are the ones to extract some waveforms to compute templates:
    * "random_spikes" : select randomly a subset of spikes per unit
    * "waveforms" : extract waveforms per unit
    * "templates": compute templates using average or median
    * "noise_levels" : compute noise levels from traces (useful to get the snr of units)
  * can be in memory or persistent to disk (2 formats binary/npy or zarr)

More extensions are available in `spikeinterface.postprocessing` like "principal_components", "spike_amplitudes",
"unit_lcations", ...


Here is the how!
"""

import matplotlib.pyplot as plt

from spikeinterface import download_dataset
from spikeinterface import create_sorting_analyzer, load_sorting_analyzer
import spikeinterface.extractors as se

##############################################################################
# First let's use the repo https://gin.g-node.org/NeuralEnsemble/ephy_testing_data
# to download a MEArec dataset. It is a simulated dataset that contains "ground truth"
# sorting information:

repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
remote_path = "mearec/mearec_test_10s.h5"
local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)

##############################################################################
# Let's now instantiate the recording and sorting objects:

recording = se.MEArecRecordingExtractor(local_path)
print(recording)
sorting = se.MEArecSortingExtractor(local_path)
print(sorting)

###############################################################################
# The MEArec dataset already contains a probe object that you can retrieve
# and plot:

probe = recording.get_probe()
print(probe)
from probeinterface.plotting import plot_probe

plot_probe(probe)

###############################################################################
# A :py:class:`~spikeinterface.core.SortingAnalyzer` object can be created with the
# :py:func:`~spikeinterface.core.create_sorting_analyzer` function (this defaults to a sparse
# representation of the waveforms)
# Here the format is "memory".

analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
print(analyzer)

###############################################################################
# A :py:class:`~spikeinterface.core.SortingAnalyzer` object can be persistant to disk
# when using format="binary_folder" or format="zarr"

folder = "analyzer_folder"
analyzer = create_sorting_analyzer(sorting=sorting,
                                   recording=recording,
                                   format="binary_folder",
                                   return_scaled=True, # this is the default to attempt to return scaled
                                   folder=folder
                                   )
print(analyzer)

# then it can be loaded back
analyzer = load_sorting_analyzer(folder)
print(analyzer)

###############################################################################
# No extensions are computed yet.
# Lets compute the most basic ones : select some random spikes per units,
# extract waveforms (sparse in this example) and compute templates.
# You can see that printing the object indicates which extension are already computed.

analyzer.compute(
    "random_spikes",
    method="uniform",
    max_spikes_per_unit=500,
)
analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
analyzer.compute("templates", operators=["average", "median", "std"])
print(analyzer)


###############################################################################
# To speed up computation, some steps like "waveforms" can also be extracted
# using parallel processing (recommended!). Like this

analyzer.compute(
    "waveforms", ms_before=1.0, ms_after=2.0, n_jobs=8, chunk_duration="1s", progress_bar=True
)

# which is equivalent to this:
job_kwargs = dict(n_jobs=8, chunk_duration="1s", progress_bar=True)
analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)

#################################################################################
# Because certain extensions rely on others (e.g. we need waveforms to calculate
# the templates) if we recompute an extension it will delete any children extensions
# Since we just recalculated "waveforms" when we print our analyzer we will see
# that it no longer has templates

print(analyzer)

# so let's get our templates back.
analyzer.compute("templates", operators=["average", "median", "std"])

###############################################################################
# Each extension can retrieve some data
# For instance the "waveforms" extension can retrieve waveforms per units
# which is a numpy array of shape (num_spikes, num_sample, num_channel):

ext_wf = analyzer.get_extension("waveforms")
for unit_id in analyzer.unit_ids:
    wfs = ext_wf.get_waveforms_one_unit(unit_id)
    print(unit_id, ":", wfs.shape)

###############################################################################
# Same for the "templates" extension. Here we can get all templates at once
# with shape (num_units, num_sample, num_channel):
# For this extension, we can get the template for all units either using the median
# or the average

ext_templates = analyzer.get_extension("templates")

av_templates = ext_templates.get_data(operator="average")
print(av_templates.shape)

median_templates = ext_templates.get_data(operator="median")
print(median_templates.shape)


###############################################################################
# This can be plotted easily.

for unit_index, unit_id in enumerate(analyzer.unit_ids[:3]):
    fig, ax = plt.subplots()
    template = av_templates[unit_index]
    ax.plot(template)
    ax.set_title(f"{unit_id}")


###############################################################################
# The SortingAnalyzer can be saved to another format using save_as()
# So the computation can be done with format="memory" and then saved to disk
# in the zarr format by using save_as()

analyzer.save_as(folder="analyzer.zarr", format="zarr")


###############################################################################
# The SortingAnalyzer also offers select_units() method which allows exporting
# only some relevant units for instance to a new SortingAnalyzer instance.

analyzer_some_units = analyzer.select_units(
    unit_ids=analyzer.unit_ids[:5], format="binary_folder", folder="analyzer_some_units"
)
print(analyzer_some_units)


plt.show()
