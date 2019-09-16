'''
RecordingExtractor objects
==========================

The :code:`RecordingExtractor` is the basic class for handling recorded data. Here is how it works.
'''

import numpy as np
import spikeinterface.extractors as se

##############################################################################
# We will create a :code:`RecordingExtractor` object from scratch using :code:`numpy` and the
# :code:`NumpyRecordingExtractor`
#
# Let's define the properties of the dataset

num_channels = 7
sampling_frequency = 30000  # in Hz
duration = 20
num_timepoints = int(sampling_frequency * duration)

##############################################################################
# We can generate a pure-noise timeseries dataset recorded by a linear probe geometry

timeseries = np.random.normal(0, 10, (num_channels, num_timepoints))
geom = np.zeros((num_channels, 2))
geom[:, 0] = range(num_channels)

##############################################################################
# And instantiate a :code:`NumpyRecordingExtractor`:

recording = se.NumpyRecordingExtractor(timeseries=timeseries, geom=geom, sampling_frequency=sampling_frequency)

##############################################################################
# We can now print properties that the RecordingExtractor retrieves from the underlying recording.

print('Num. channels = {}'.format(len(recording.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(recording.get_traces(channel_ids=2))))
print('Location of third electrode = {}'.format(recording.get_channel_property(channel_id=2, property_name='location')))

##############################################################################
# Some extractors also implement a :code:`write` function. We can for example save our newly created recording into
# MDA format (Mountainsort4 format):

se.MdaRecordingExtractor.write_recording(recording=recording, save_path='sample_mountainsort_dataset')

##############################################################################
# and read it back with the proper extractor:

recording2 = se.MdaRecordingExtractor(folder_path='sample_mountainsort_dataset')
print('Num. channels = {}'.format(len(recording2.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording2.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording2.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(recording2.get_traces(channel_ids=2))))
print('Location of third electrode = {}'.format(recording.get_channel_property(channel_id=2, property_name='location')))

##############################################################################
# Sometimes experiments are run with different conditions, e.g. a drug is applied, or stimulation is performed. 
# In order to define different phases of an experiment, one can use epochs:

recording2.add_epoch(epoch_name='stimulation', start_frame=1000, end_frame=6000)
recording2.add_epoch(epoch_name='post_stimulation', start_frame=6000, end_frame=10000)
recording2.add_epoch(epoch_name='pre_stimulation', start_frame=0, end_frame=1000)

recording2.get_epoch_names()

##############################################################################
# An Epoch can be retrieved and it is returned as a :code:`SubRecordingExtractor`, which is a subclass of the
# :code:`RecordingExtractor`, hence maintaining the same functionality.

recording3 = recording2.get_epoch(epoch_name='stimulation')
epoch_info = recording2.get_epoch_info('stimulation')
start_frame = epoch_info['start_frame']
end_frame = epoch_info['end_frame']

print('Epoch Name = stimulation')
print('Start Frame = {}'.format(start_frame))
print('End Frame = {}'.format(end_frame))
print('Mean. on second channel during stimulation = {}'.format(np.mean(recording3.get_traces(channel_ids=1))))
print('Location of third electrode = {}'.format(recording.get_channel_property(channel_id=2, property_name='location')))

##############################################################################
# :code:`SubRecordingExtractor` objects can be used to extract arbitrary subsets of your data/channels manually without
# epoch functionality:

recording4 = se.SubRecordingExtractor(parent_recording=recording2, channel_ids=[2, 3, 4, 5], start_frame=14000,
                                      end_frame=16000)

print('Num. channels = {}'.format(len(recording4.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording4.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording4.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(recording4.get_traces(channel_ids=2))))
print(
    'Location of third electrode = {}'.format(recording4.get_channel_property(channel_id=2, property_name='location')))

##############################################################################
# or to remap the channel ids:

recording5 = se.SubRecordingExtractor(parent_recording=recording2, channel_ids=[2, 3, 4, 5],
                                      renamed_channel_ids=[0, 1, 2, 3],
                                      start_frame=14000, end_frame=16000)
print('New ids = {}'.format(recording5.get_channel_ids()))
print('Original ids = {}'.format(recording5.get_original_channel_ids([0, 1, 2, 3])))
print('Num. channels = {}'.format(len(recording5.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording5.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording5.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(recording5.get_traces(channel_ids=0))))
print(
    'Location of third electrode = {}'.format(recording5.get_channel_property(channel_id=0, property_name='location')))
