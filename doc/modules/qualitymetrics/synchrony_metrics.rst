Synchrony Metrics (:code:`synchrony_metrics`)
=======================================

Calculation
-----------
This function is providing a metric for the presence of synchronous spiking events across multiple spike trains.

The complexity is used to characterize synchronous events within the same spike train and across different spike
trains. This way synchronous events can be found both in multi-unit and single-unit spike trains.

Complexity is calculated by counting the number of spikes (i.e. non-empty bins) that occur separated by spread - 1 or less empty bins,
within and across spike trains in the spiketrains list.

Expectation and use
-------------------
A larger value indicates a higher synchrony of the respective spike train with the other spike trains.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as qm
    # Make recording, sorting and wvf_extractor object for your data.
    presence_ratio = qm.compute_synchrony_metrics(wvf_extractor)
    # presence_ratio is a tuple of dicts with the synchrony metrics for each unit

Links to source code
--------------------

From `Elephant - Electrophysiology Analysis Toolkit <https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_synchrony.py#L245>`_


References
----------

.. automodule:: spikeinterface.toolkit.qualitymetrics.misc_metrics

    .. autofunction:: compute_synchrony_metrics

Literature
----------

Described in Gruen_

Citations
---------
.. [Gruen] Sonja Grün, Moshe Abeles, and Markus Diesmann. Impact of higher-order correlations on coincidence distributions of massively parallel data.
In International School on Neural Networks, Initiated by IIASS and EMFCSC, volume 5286, 96–114. Springer, 2007.
