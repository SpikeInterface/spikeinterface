Firing range (:code:`firing_range`)
===================================


Calculation
-----------

The firing range indicates the spread of the firing range of a unit across the recording. It is computed by
taking the difference between the 95-th and 5th percentiles firing rates computed over short time bins (e.g. 10 s).



Expectation and use
-------------------

Both very high and very low firing rates can indicate errors.
Highly contaminated units (type I error) may have high firing rates as a result of the inclusion of other neurons' spikes.
Low firing rate units are likely to be incomplete (type II error), although this is not always the case (some neurons have highly selective firing patterns).
The firing rate is expected to be approximately log-normally distributed [Buzsáki]_.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as qm

    # Make recording, sorting and wvf_extractor object for your data.
    firing_rate = qm.compute_firing_ranges(wvf_extractor)
    # firing_rate is a dict containing the units' IDs as keys,
    # and their firing rates across segments as values (in Hz).

References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_firing_rates


Links to original implementations
---------------------------------

* From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Firing-rate>`_

Literature
----------

Unknown origin.
Widely discussed eg: [Buzsáki]_.
