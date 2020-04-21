"""
Validation Tutorial
======================

After spike sorting, you might want to validate the goodness of the sorted units. This can be done using the
:code:`toolkit.validation` submodule, which computes several quality metrics of the sorted units.

"""

import spikeinterface.extractors as se
import spikeinterface.toolkit as st

##############################################################################
# First, let's create a toy example:

recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=10, seed=0)

##############################################################################
# The :code:`toolkit.validation` submodule has a set of functions that allow users to compute metrics in a
# compact and easy way. To compute a single metric, the user can simply run one of the quality metric functions
# as shown below. Each function as a variety of adjustable parameters that can be tuned by the user to match their data.

firing_rates = st.validation.compute_firing_rates(sorting, duration_in_frames=recording.get_num_frames())
isi_violations = st.validation.compute_isi_violations(sorting, duration_in_frames=recording.get_num_frames(), isi_threshold=0.0015)
snrs = st.validation.compute_snrs(recording=recording, sorting=sorting, max_spikes_per_unit_for_snr=1000)
nn_hit_rate, nn_miss_rate = st.validation.compute_nn_metrics(recording=recording, sorting=sorting, num_channels_to_compare=13)

##############################################################################
# To compute more than one metric at once, a user can use the :code:`compute_quality_metrics` function and indicate
# which metrics they want to compute. This will return a dictionary of metrics or optionally a pandas dataframe.
metrics = st.validation.compute_quality_metrics(sorting=sorting, recording=recording, 
                                                metric_names=['firing_rate', 'isi_viol', 'snr', 'nn_hit_rate', 'nn_miss_rate'], 
                                                as_dataframe=True)

##############################################################################
# To compute metrics on only part of the recording, a user can specify specific epochs in the Recording and Sorting extractor
# using :code:`add_epoch` and then compute the metrics on the SubRecording and SubSorting extractor given by :code:`get_epoch`.
# In this example, we compute all the same metrics on the first half of the recording.
sorting.add_epoch(epoch_name="first_half", start_frame=0, end_frame=recording.get_num_frames()/2) #set
recording.add_epoch(epoch_name="first_half", start_frame=0, end_frame=recording.get_num_frames()/2)
subsorting = sorting.get_epoch("first_half")
subrecording = recording.get_epoch("first_half")
metrics_first_half = st.validation.compute_quality_metrics(sorting=subsorting, recording=subrecording, 
                                                           metric_names=['firing_rate', 'isi_viol', 'snr', 'nn_hit_rate', 'nn_miss_rate'], 
                                                           as_dataframe=True)

print("Metrics full recording")
print(metrics)
print('\n')
print("Metrics first half recording")
print(metrics_first_half)