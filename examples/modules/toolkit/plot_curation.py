"""
Curation Tutorial
======================

After spike sorting and computing validation metrics, you can automatically curate the spike sorting output using the
quality metrics. This can be done with the :code:`toolkit.curation` submodule.

"""

import spikeinterface.extractors as se
import spikeinterface.toolkit as st

##############################################################################
# First, let's create a toy example:

recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)
