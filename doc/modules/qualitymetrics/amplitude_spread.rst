Amplitude spread (:code:`amplitude_spread`)
===========================================


Calculation
-----------

The amplitude spread is a measure of the amplitude variability.
It is computed as the ratio between the standard deviation and the amplitude mean (aka the coefficient of variation).
To obtain a better estimate of this measure, it is first computed separately for several bins of a prefixed number of spikes
(e.g. 100) and then the median of these values is taken.

The computation requires either spike amplitudes (see :py:func:`~spikeinterface.postprocessing.compute_spike_amplitudes()`)
or amplitude scalings (see :py:func:`~spikeinterface.postprocessing.compute_amplitude_scalings()`) to be pre-computed.


Expectation and use
-------------------

Very high levels of amplitude_spread ranges, outside of a physiolocigal range, might indicate noise contamination.


Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as qm

    # Make recording, sorting and wvf_extractor object for your data.
	# It is required to run `compute_spike_amplitudes(wvf_extractor)` or
	# `compute_amplitude_scalings(wvf_extractor)` (if missing, values will be NaN)
    amplitude_spread = qm.compute_firing_ranges(wvf_extractor, amplitude_extension='spike_amplitudes')
    # amplitude_spread is a dict containing the units' IDs as keys,
    # and their amplitude_spread (in units of standard deviation).



References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_amplitude_spreads


Literature
----------

Designed by Simon Musall and adapted to SpikeInterface by Alessio Buccino.
