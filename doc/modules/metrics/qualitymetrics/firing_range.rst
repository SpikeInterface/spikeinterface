Firing range (:code:`firing_range`)
===================================


Calculation
-----------

The firing range indicates the dispersion of the firing rate of a unit across the recording. It is computed by
taking the difference between the 95th percentile's firing rate and the 5th percentile's firing rate computed over short time bins (e.g. 10 s).



Expectation and use
-------------------

Very high levels of firing ranges, outside of a physiological range, might indicate noise contamination.


Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    # Combine a sorting and recording into a sorting_analyzer
    firing_range = sqm.compute_firing_ranges(sorting_analyzer=sorting_analyzer)
    # firing_range is a dict containing the unit IDs as keys,
    # and their firing firing_range as values (in Hz).

References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_firing_ranges


Literature
----------

Designed by Simon Musall and adapted to SpikeInterface by Alessio Buccino.
