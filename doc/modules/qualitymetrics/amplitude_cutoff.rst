.. _amp_cutoff:

Amplitude cutoff (:code:`amplitude_cutoff`)
===========================================

Calculation
-----------

A histogram of spike amplitudes is created and deviations from the expected symmetrical distribution are identified.

Expectation and use
-------------------

Deviations from the expected Gaussian distribution are used to estimate the number of spikes missing from the unit.
This yields an estimate of the number of spikes missing from the unit (false negative rate).
A smaller value for this metric is preferred, as this indicates fewer false negatives.
The distribution can be computed on chunks for larger recording, as drift can impact the spike amplitudes (and thus not give a Gaussian distribution anymore).

Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as sqm

	# It is also recommended to run `compute_spike_amplitudes(wvf_extractor)`
	# in order to use amplitudes from all spikes
	fraction_missing = sqm.compute_amplitude_cutoffs(wvf_extractor, peak_sign="neg")
	# fraction_missing is a dict containing the unit IDs as keys,
	# and their estimated fraction of missing spikes as values.

Reference
---------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_amplitude_cutoffs


Links to original implementations
---------------------------------

* From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L219/>`_

* From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Amplitude-cutoff>`_

Literature
----------

Introduced by [Hill]_.
