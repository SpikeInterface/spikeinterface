Noise cutoff (not currently implemented)
========================================

Calculation
-----------


Metric describing whether an amplitude distribution is cut off, similar to _amp_cutoff  :ref:`amplitude cutoff <amp_cutoff>` but without a Gaussian assumption. 
A histogram of amplitudes is created and quantifies the distance between the low tail, mean number of spikes and high tail in terms of standard deviations.

A SpikeInterface implementation is not yet available.

Expectation and use
-------------------

Noise cutoff attempts to describe whether an amplitude distribution is cut off.
The metric is loosely based on [Hill]_'s amplitude cutoff, but is here adapted (originally by [IBL]_) to avoid making the Gaussianity assumption on spike distributions.
Noise cutoff provides an estimate of false negative rate, so a lower value indicates fewer missed spikes (a more complete unit).


Links to original implementations
---------------------------------

* From `IBL implementation <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_


Literature
----------

Metric introduced by [IBL]_ (adapted from [Hill]_'s amplitude cutoff metric).

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.

.. [IBL] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022.



