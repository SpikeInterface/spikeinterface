.. _amp_median:

Amplitude median (:code:`amplitude_median`)
===========================================

Calculation
-----------

Geometric median amplitude is computed in the log domain.
The metric is then converted back to original units.

Expectation and use
-------------------

A larger value (larger signal) indicates a better unit.


Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as sqm

	# It is also recommended to run `compute_spike_amplitudes(wvf_extractor)`
	# in order to use amplitude values from all spikes.
	amplitude_medians = sqm.compute_amplitude_medians(waveform_extractor=wvf_extractor)
	# amplitude_medians is a dict containing the unit IDs as keys,
	# and their estimated amplitude medians as values.

Reference
---------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_amplitude_medians


Links to original implementations
---------------------------------

* From `IBL <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_


Literature
----------

Introduced by [IBL]_.
