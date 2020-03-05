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
# as shown below.

snrs = st.validation.compute_snrs(recording=recording, sorting=sorting)
##############################################################################
# Each quality metric function has a variety of adjustable parameters. Any parameters that end with :code:`_params` 
# can be adjusted by passing in dictionaries with new values.
feature_params  = {'max_spikes_per_unit':400}
snrs = st.validation.compute_snrs(recording=recording, sorting=sorting, feature_params=feature_params)
##############################################################################
# To compute more than one metric at once, a user can use the :code:`compute_metrics` function and indicate
# which parameters they want to compute. This will return a dictionary of metrics.
metrics = st.validation.compute_metrics(sorting=sorting, recording=recording, metric_names=['snr', 'isolation_distance']) 

##############################################################################
# Metrics can also be computed over muliple epochs of time by adjusting the :code:`epoch_params` 
metrics = st.validation.compute_metrics(sorting=sorting, recording=recording, epoch_params={'epoch_tuples':[(0,5),(5,10)]}, 
                                        metric_names=["amplitude_cutoff", "firing_rate"], save_as_property=False)
