Amplitude median (not yet implemented)
======================================

Calculation
-----------

Geometric median amplitude is computed in the log domain.
In the original [IBL]_ implementation, the metric is converted back to volt units.

A SpikeInterface implementation is not yet available.

Expectation and use
-------------------

A larger value (larger signal) is taken to indicate a better unit.

Literature
----------

Metric introduced by IBL_.

.. [IBL] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022. 


Links to source code
--------------------

`IBL implementation <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_
