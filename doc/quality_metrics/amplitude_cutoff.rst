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

	import spikeinterface.toolkit as st

	# Make recording, sorting and wvf_extractor objects for your data.
	
	fraction_missing = st.compute_amplitudes_cutoff(wvf_extractor, peak_sign="neg")
	# fraction_missing is a dict containing the units' ID as keys,
	# and their estimated fraction of missing spikes as values.

Reference
---------

.. automodule:: spikeinterface.toolkit.qualitymetrics.misc_metrics

	.. autofunction:: compute_amplitudes_cutoff


Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/ae679aff788a6dd4d8e7783e1f72ec7e550c1bf9/spikeinterface/toolkit/qualitymetrics/misc_metrics.py#L259/>`_

From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L219/>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Amplitude-cutoff>`_

Literature
----------

Introduced by [Hill]_.

Citations
---------

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.