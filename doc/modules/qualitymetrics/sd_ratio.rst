Standard Deviation (SD) ratio (:code:`sd_ratio`)
================================================

Calculation
-----------

All spikes from the same neuron should have the same shape. This means that at the peak of the spike, the standard deviation of the voltage should be the same as that of noise. If spikes from multiple neurons are grouped into a single unit, the standard deviation of spike amplitudes would likely be increased.

This metric, first described [Pouzat]_ then adapted by Wyngaard, Llobet & Barbour (in preparation), returns the ratio between both standard deviations:

.. math::
	S = \frac{\sigma_{\mathrm{unit}}}{\sigma_{\mathrm{noise}}}

To remove the effect of drift on spikes amplitude, :math:`\sigma_{\mathrm{unit}}` is computed by subtracting each spike amplitude, and dividing the resulting standard deviation by :math:`\sqrt{2}`.
Also to remove the effect of bursts (which can have lower amplitudes), you can specify a censored period (by default 4.0 ms) where spikes happening less than this period after another spike will not be considered.


Expectation and use
-------------------

For a unit representing a single neuron, this metric should return a value close to one. However for units that are contaminated, the value can be significantly higher.


Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as sqm

	sd_ratio = sqm.compute_sd_ratio(wvf_extractor, censored_period_ms=4.0)


Literature
----------

Introduced by [Pouzat]_ (2002).
Expanded by Wyngaard, Llobet and Barbour (in preparation).
