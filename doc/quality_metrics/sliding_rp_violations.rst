Sliding refractory period violations (not yet implemented)
==========================================================

Calculation
-----------

Compute maximum allowed refractory period violations for all possible refractory periods in recording.
Bins of 0.25ms are used in the [IBL]_ implementation.
For each bin, this maximum value is compared with that observed in the recording.
In the IBL_ implementation, a threshold is imposed and a binary value returned (based on whether the unit 'passes' the metric).
A SpikeInterface implementation is in development.

Expectation and use
-------------------

Similar to the :ref:`ISI violations <ISI>` metric, this metric quantifies the number of refractory period violations seen in the recording of the unit.
This is an estimate of false positive rate.
A high number of violations indicates contamination, so a low value is expected for true units.


Literature
----------

Metric introduced by IBL_.

.. [IBL] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022. 

Links to source code
--------------------

`IBL implementation <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_
