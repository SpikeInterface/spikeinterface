Firing rate
===========



Calculation
-----------

Firing rate is simply the average number of spikes within the recording per second.

.. math::
    \textrm{firing rate} = \frac{N_s}{T_r}

- :math:`T_r` : duration of recording in seconds (:code:`total_duration` in SpikeInterface).
- :math:`N_s` : number of spikes observed in full recording (:code:`n` in SpikeInterface).

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

    import spikeinterface.toolkit as st

    # Make recording, sorting and wvf_extractor object for your data.
    
    firing_rate = st.compute_firing_rate(wvf_extractor)
    # firing_rate is a dict containing the units' ID as keys,
    # and their firing rate across segments as values (in Hz).

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/ae679aff788a6dd4d8e7783e1f72ec7e550c1bf9/spikeinterface/toolkit/qualitymetrics/misc_metrics.py#L52>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Firing-rate>`_

Literature
----------

Unknown origin.
Widely discussed eg: Buzsáki_.

Citations
---------

.. [Buzsáki] Buzsáki, György, and Kenji Mizuseki. “The Log-Dynamic Brain: How Skewed Distributions Affect Network Operations.” Nature reviews. Neuroscience 15.4 (2014): 264–278. Web.