Firing rate (:code:`firing_rate`)
=================================


Calculation
-----------

Firing rate is simply the average number of spikes within the recording per second.

.. math::
    \textrm{firing rate} = \frac{N_s}{T_r}

- :math:`N_s` : number of spikes observed in full recording.
- :math:`T_r` : duration of recording in seconds.

Expectation and use
-------------------

Both very high and very low values of firing rate can indicate errors.
Highly contaminated units (type I error) may have high firing rates as a result of inclusion of other neurons' spikes.
Low firing rate units are likely to be incomplete (type II error), although this is not always the case (some neurons have highly selective firing patterns).
The firing rate is expected to be approximately log-normally distributed [Buzsáki]_.

Example code
------------

Without SpikeInterface:

.. code-block:: python
    
    spike_train = ...
    t_recording = ...    # Length of recording (in s).

    firing_rate = len(spike_train) / t_recording

With SpikeInterface:

.. code-block:: python

    import spikeinterface.qualitymetrics as qm

    # Make recording, sorting and wvf_extractor object for your data.
    firing_rate = qm.compute_firing_rates(wvf_extractor)
    # firing_rate is a dict containing the units' ID as keys,
    # and their firing rate across segments as values (in Hz).

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
