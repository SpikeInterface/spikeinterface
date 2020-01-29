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
# The :code:`toolkit.validation` submodule has a :code:`MetricCalculator` class that enables to compute metrics in a
# compact and easy way. You first need to instantiate a :code:`MetricCalculator` object with the
# :code:`SortingExtractor` and :code:`RecordingExtractor` objects.

mc = st.validation.MetricCalculator(sorting, recording)

##############################################################################
# You can then compute metrics as follows:

mc.compute_metrics()

##############################################################################
# This is the list of the computed metrics:
print(list(mc.get_metrics_dict().keys()))

##############################################################################
# The :code:`get_metrics_dict` and :code:`get_metrics_df` return all metrics as a dictionary or a pandas dataframe:

print(mc.get_metrics_dict())
print(mc.get_metrics_df())


##############################################################################
# If you don't need to compute all metrics, you can either pass a 'metric_names' list to the :code:`compute_metrics` or
# call separate methods for computing single metrics:

# This only compute signal-to-noise ratio (SNR)
mc.compute_metrics(metric_names=['snr'])
print(mc.get_metrics_df()['snr'])

# This function also returns the SNRs
snrs = st.validation.compute_snrs(sorting, recording)
print(snrs)
