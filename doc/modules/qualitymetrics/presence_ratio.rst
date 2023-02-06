Presence ratio (:code:`presence_ratio`)
=======================================

Calculation
-----------

Presence ratio is the proportion of discrete time bins in which at least one spike occurred.

.. math::
    \textrm{presence ratio} = \frac{B_s}{B_s + B_n}

- :math:`B_s` : number of bins in which spikes occurred.
- :math:`B_n` : number of bins in which no spikes occurred.

Expectation and use
-------------------

Complete units are expected to have a presence ratio of 90% or more.
Low presence ratio (close to 0) can indicate incompleteness (type II error) or highly selective firing pattern.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as qm

    # Make recording, sorting and wvf_extractor object for your data.

    presence_ratio = qm.compute_presence_ratio(wvf_extractor)
    # presence_ratio is a dict containing the units' ID as keys
    # and their presence ratio (between 0 and 1) as values.

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/qualitymetrics/misc_metrics.py#L87>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Presence-ratio>`_


References
----------

.. automodule:: spikeinterface.toolkit.qualitymetrics.misc_metrics

    .. autofunction:: compute_presence_ratios

Literature
----------

Unknown origin.

Citations
---------

