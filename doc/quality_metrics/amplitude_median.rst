Amplitude median
================

Calculation
-----------

Geometric median amplitude is computed in the log domain.
The metric is then converted back to original units.

Expectation and use
-------------------

A larger value (larger signal) is taken to indicate a better unit.


Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as qm

	# Make recording, sorting and wvf_extractor objects for your data.
    # It is also recommended to run `compute_spike_amplitudes(wvf_extractor)`
    # in order to use amplitude values from all spikes.

	amplitude_medians = qm.compute_amplitudes_median(wvf_extractor)
	# amplitude_medians is a dict containing the units' ID as keys,
	# and their estimated amplitude medians as values.

Literature
----------

Metric introduced by IBL_.

.. [IBL] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022. 


Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/qualitymetrics/misc_metrics.py#L491/>`_

From `IBL <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_
