Sliding refractory period violations (:code:`sliding_rp_violations`)
====================================================================

Calculation
-----------

Compute maximum allowed refractory period violations for all possible refractory periods in recording.
Bins of 0.25ms are used in the [IBL]_ implementation.
For each bin, this maximum value is compared with that observed in the recording.

In the [IBL]_ implementation, a threshold is imposed and a binary value returned (based on whether the unit 'passes' the metric).
The SpikeInterface implementation, instead, returns the minimum contamination with at least 90% confidence.
This contamination value is between 0 and 1.

Expectation and use
-------------------

Similar to the :ref:`ISI violations <ISI>` metric, this metric quantifies the number of refractory period violations seen in the recording of the unit.
This is an estimate of the false positive rate.
A high number of violations indicates contamination, so a low value is expected for high quality units.


Example code
------------

With SpikeInterface:

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    # Make recording, sorting and wvf_extractor object for your data.

    contamination = sqm.compute_sliding_rp_violations(waveform_extractor=wvf_extractor, bin_size_ms=0.25)

References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_sliding_rp_violations


Links to original implementations
---------------------------------

* From `IBL <https://github.com/int-brain-lab/ibllib/blob/2e1f91c622ba8dbd04fc53946c185c99451ce5d6/brainbox/metrics/single_units.py>`_

* From `Steinmetz Lab <https://github.com/SteinmetzLab/slidingRefractory/blob/1.0.0/python/slidingRP/metrics.py>`_

Literature
----------

Metric introduced by [IBL]_.
