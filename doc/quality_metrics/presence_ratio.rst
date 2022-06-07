Presence ratio
==============

Calculation
-----------

Presence ratio is the proportion of discrete time bins in which at least one spike occurred.

.. math::
    \textrm{presence ratio} = \frac{B_s}{B_s + B_n}

- :math:`B_s` : number of bins in which spikes occurred (in SpikeInterface).
- :math:`B_n` : number of bins in which no spikes occurred (in SpikeInterface).

Expectation and use
-------------------

Complete units are expected to have a presence ratio of 90% or more.
Low presence ratio (close to 0) can indicate incompleteness (type II error) or highly selective firing pattern.

Example code
------------

.. code-block:: python

    import spiketoolkit as st

    # Make recording, sorting and wvf_extractor object for your data.

    presence_ratio = st.compute_presence_ratio(wvf_extractor)
    # presence_ratio is a dict containing the units' ID as keys
    # and their presence ratio (between 0 and 1) as values.

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/85244cd686bfe2a80649246ea1e29120930cb9c7/spikeinterface/toolkit/qualitymetrics/misc_metrics.py#L87>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Presence-ratio>`_

Literature
----------

Unknown origin.

Citations
---------

