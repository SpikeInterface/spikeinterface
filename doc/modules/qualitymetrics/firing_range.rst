Firing range (:code:`firing_range`)
===================================


Calculation
-----------

The firing range indicates the dispersion of the firing rate of a unit across the recording. It is computed by
taking the difference between the 95-th and 5th percentiles firing rates computed over short time bins (e.g. 10 s).



Expectation and use
-------------------

Very high levels of firing ranges, outside of a physiological range, might indicate noise contamination.


Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as qm

    # Make recording, sorting and wvf_extractor object for your data.
    firing_range = qm.compute_firing_ranges(wvf_extractor)
    # firing_range is a dict containing the units' IDs as keys,
    # and their firing firing_range as values (in Hz).

References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_firing_ranges


Literature
----------

Designed by Simon Musall and adapted to SpikeInterface by Alessio Buccino.
