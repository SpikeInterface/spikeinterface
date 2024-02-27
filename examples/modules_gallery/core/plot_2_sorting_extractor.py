"""
Sorting objects
===============

The :py:class:`~spikeinterface.core.BaseSorting` is the basic class for handling spike sorted data.
Here is how it works.

A SortingExtractor handles:

  * spike trains retrieval across segments
  * dumping to/loading from dict-json
  * saving (caching)

"""

import numpy as np
import spikeinterface.extractors as se

##############################################################################
# We will create a :code:`SortingExtractor` object from scratch using :code:`numpy` and the
# :py:class:`~spikeinterface.core.NumpySorting`
#
# Let's define the properties of the dataset:

sampling_frequency = 30000.0
duration = 20.0
num_timepoints = int(sampling_frequency * duration)
num_units = 4
num_spikes = 1000

##############################################################################
# We generate some random events for 2 segments:

times0 = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
labels0 = np.random.randint(1, num_units + 1, size=num_spikes)

times1 = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_spikes)))
labels1 = np.random.randint(1, num_units + 1, size=num_spikes)

##############################################################################
# And instantiate a :py:class:`~spikeinterface.core.NumpySorting` object:

sorting = se.NumpySorting.from_times_labels([times0, times1], [labels0, labels1], sampling_frequency)
print(sorting)

##############################################################################
# We can now print properties that the :code:`SortingExtractor` retrieves from
# the underlying sorted dataset.

print("Unit ids = {}".format(sorting.get_unit_ids()))
st = sorting.get_unit_spike_train(unit_id=1, segment_index=0)
print("Num. events for unit 1seg0 = {}".format(len(st)))
st1 = sorting.get_unit_spike_train(unit_id=1, start_frame=0, end_frame=30000, segment_index=1)
print("Num. events for first second of unit 1 seg1 = {}".format(len(st1)))

##############################################################################
# Some extractors also implement a :code:`write` function. We can for example
# save our newly created sorting object to NPZ format (a simple format based
# on numpy used in :code:`spikeinterface`):

file_path = "my_sorting.npz"
se.NpzSortingExtractor.write_sorting(sorting, file_path)

##############################################################################
# We can now read it back with the proper extractor:

sorting2 = se.NpzSortingExtractor(file_path)
print(sorting2)

##############################################################################
# Unit properties are key value pairs that we can store for any unit.
# We will now calculate unit firing rates and add them as properties to
# the :code:`SortingExtractor` object:

firing_rates = []
for unit_id in sorting2.get_unit_ids():
    st = sorting2.get_unit_spike_train(unit_id=unit_id, segment_index=0)
    firing_rates.append(st.size / duration)
sorting2.set_property("firing_rate", firing_rates)

print(sorting2.get_property("firing_rate"))

##############################################################################
# You can also get a a sorting with a subset of unit. Properties are
# propagated to the new object:

sorting3 = sorting2.select_units(unit_ids=[1, 4])
print(sorting3)

print(sorting3.get_property("firing_rate"))

# which is equivalent to
from spikeinterface import UnitsSelectionSorting

sorting3 = UnitsSelectionSorting(sorting2, unit_ids=[1, 4])

###############################################################################
# A sorting can be "dumped" (exported) to:
#  * a dict
#  * a json file
#  * a pickle file
#
# The "dump" operation is lazy, i.e., the spike trains are not exported.
# Only the information about how to reconstruct the sorting are dumped:

from spikeinterface import load_extractor
from pprint import pprint

d = sorting2.to_dict()
pprint(d)

sorting2_loaded = load_extractor(d)
print(sorting2_loaded)

###############################################################################
# The dictionary can also be dumped directly to a JSON file on disk:

sorting2.dump("my_sorting.json")

sorting2_loaded = load_extractor("my_sorting.json")
print(sorting2_loaded)

###############################################################################
# **IMPORTANT**: the "dump" operation DOES NOT copy the spike trains to disk!
#
# If you wish to also store the spike trains in a compact way you need to use the
# :code:`save()` function:


sorting2.save(folder="./my_sorting")

import os

pprint(os.listdir("./my_sorting"))

sorting2_cached = load_extractor("./my_sorting")
print(sorting2_cached)
