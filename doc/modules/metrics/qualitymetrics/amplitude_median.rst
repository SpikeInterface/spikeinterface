.. _amp_median:

Amplitude median (:code:`amplitude_median`)
===========================================

Calculation
-----------

Computed simply as the median of the amplitudes for each unit.

Expectation and use
-------------------

A larger value (larger signal) indicates a better unit.


Example code
------------

.. code-block:: python

	import spikeinterface.metrics.quality as sqm

	# It is also recommended to run sorting_analyzer.compute(input="spike_amplitudes")
	# in order to use amplitude values from all spikes.
	amplitude_medians = sqm.compute_amplitude_medians(sorting_analyzer)
	# amplitude_medians is a dict containing the unit IDs as keys,
	# and their estimated amplitude medians as values.

Reference
---------

.. autofunction:: spikeinterface.metrics.quality.misc_metrics.compute_amplitude_medians


Links to original implementations
---------------------------------

* From `IBL <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_


Literature
----------

Introduced by [IBL]_.
