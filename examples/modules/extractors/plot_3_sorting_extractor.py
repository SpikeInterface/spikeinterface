'''
SortingExtractor objects
========================

The :code:`SortingExtractor` is the basic class for handling spike sorted data. Here is how it works.
'''

import numpy as np
import spikeinterface.extractors as se

##############################################################################
# We will create a :code:`SortingExtractor` object from scratch using :code:`numpy` and the
# :code:`NumpySortingExtractor`
#
# Let's define the properties of the dataset

samplerate = 30000
duration = 20
num_timepoints = int(samplerate * duration)
num_units = 4
num_events = 1000

##############################################################################
# We generatesome random events.

times = np.int_(np.sort(np.random.uniform(0, num_timepoints, num_events)))
labels = np.random.randint(1, num_units + 1, size=num_events)

##############################################################################
# And instantiate a :code:`NumpyRecordingExtractor`:

sorting = se.NumpySortingExtractor()
sorting.set_times_labels(times=times, labels=labels)
sorting.set_sampling_frequency(sampling_frequency=samplerate)

##############################################################################
# We can now print properties that the SortingExtractor retrieves from the underlying sorted dataset.

print('Unit ids = {}'.format(sorting.get_unit_ids()))
st = sorting.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sorting.get_unit_spike_train(unit_id=1, start_frame=0, end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))

##############################################################################
# Some extractors also implement a :code:`write` function. We can for example save our newly created sorting into
# MDA format (Mountainsort4 format):

se.MdaSortingExtractor.write_sorting(sorting=sorting, save_path='firings_true.mda')

##############################################################################
# and read it back with the proper extractor:

sorting2 = se.MdaSortingExtractor(firings_file='firings_true.mda',
                                  sampling_frequency=samplerate)
print('Unit ids = {}'.format(sorting2.get_unit_ids()))
st = sorting2.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1 = sorting2.get_unit_spike_train(unit_id=1, start_frame=0, end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))

##############################################################################
# Unit properties are name value pairs that we can store for any unit. We will now calculate a unit
# property and store it in the SortingExtractor

full_spike_train = sorting2.get_unit_spike_train(unit_id=1)
firing_rate = float(len(full_spike_train)) / sorting2.get_sampling_frequency()
sorting2.set_unit_property(unit_id=1, property_name='firing_rate', value=firing_rate)
print('Average firing rate during the recording of unit 1 = {}'.format(sorting2.get_unit_property(unit_id=1,
                                                                                                  property_name='firing_rate')))
print("Spike property names: " + str(sorting2.get_shared_unit_property_names()))

##############################################################################
# :code:`SubSortingExtractor` objects can be used to extract arbitrary subsets of your units/spike trains manually

sorting3 = se.SubSortingExtractor(parent_sorting=sorting2, unit_ids=[1, 2],
                                  start_frame=10000, end_frame=20000)
print('Num. units = {}'.format(len(sorting3.get_unit_ids())))
print('Average firing rate of units1 during frames 10000-20000 = {}'.format(
    float(len(sorting3.get_unit_spike_train(unit_id=1))) / 6000))

##############################################################################
# Unit features are name value pairs that we can store for each spike. Let's load a randomly generated 'random_value'
# features. Features are used, for example, to store waveforms, amplitude, and PCA scores

random_values = np.random.randn(len(sorting3.get_unit_spike_train(unit_id=1)))
sorting3.set_unit_spike_features(unit_id=1, feature_name='random_value',
                                 value=random_values)
print("Spike feature names: " + str(sorting3.get_shared_unit_spike_feature_names()))
