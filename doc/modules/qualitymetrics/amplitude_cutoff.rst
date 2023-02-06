.. _amp_cutoff:

Amplitude cutoff (:code:`amplitude_cutoff`)
===========================================

Calculation
-----------

A histogram of spike amplitudes is created and deviations from the expected symmetrical distribution are identified.

Expectation and use
-------------------

Deviations from the expected Gaussian distributions are used to estimate the number of spikes missing from the unit.
This yields an estimate of the number of spikes missing from the unit (false negative rate).
A smaller value for this metric is preferred, as this indicates few false negatives.
The distributions can be computed on chunks for larger recording, as drift can impact the spike amplitudes (and thus not give a Gaussian distribution anymore).

Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as qm

	# Make recording, sorting and wvf_extractor objects for your data.
	# It is also recommended to run `compute_spike_amplitudes(wvf_extractor)`
    # in order to use amplitude values from all spikes.

	fraction_missing = qm.compute_amplitudes_cutoff(wvf_extractor, peak_sign="neg")
	# fraction_missing is a dict containing the units' ID as keys,
	# and their estimated fraction of missing spikes as values.

Reference
---------

.. automodule:: spikeinterface.qualitymetrics.misc_metrics

	.. autofunction:: compute_amplitudes_cutoff


Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/qualitymetrics/misc_metrics.py#L259/>`_

From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L219/>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Amplitude-cutoff>`_

Literature
----------

Introduced by [Hill]_.

Citations
---------

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.
