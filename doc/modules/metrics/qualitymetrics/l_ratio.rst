L-ratio (:code:`l_ratio`)
=========================

Calculation
-----------

This example assumes use of a tetrode.

L-ratio uses 4 principal components (PCs) for each tetrode channel (the first being energy, the square root of the sum of squares of each sample in the waveform, followed by the first 3 PCs of the energy normalised waveform).
This yields spikes which are each represented as a point in 16 dimensional space.

Define, for each cluster :math:`C`, :math:`D_{i,C}^2`, the squared Mahalanobis distance from the centre of cluster :math:`C` for every spike :math:`i` in the dataset (similarly to the calculation for isolation distance above).
Assume that spikes in the cluster distribute normally in each dimension, so that :math:`D^2` for spikes in a cluster will distribute as :math:`\chi^2` with 16 degrees of freedom.
This yields :math:`\textrm{CDF}_{\chi^2_{\mathrm{df}}}`, the cumulative distribution function of the :math:`\chi^2` distribution.
Define for each cluster :math:`C`, the value :math:`L(C)`, representing the amount of contamination of the cluster :math:`C``:

.. math::
    L(C) = \sum_{i \notin \mathrm{C}} 1 - \mathrm{CDF}_{\chi^2_{\mathrm{df}}}(D^2_{i, C})


:math:`L` is then the sum of probabilities that each spike which is not a member of cluster :math:`C` should be.
Therefore the inverse of this cumulative distribution yields the probability of cluster membership for each spike :math:`i`.
:math:`L` is then normalised by the number of spikes :math:`N_s` in :math:`C` to allow larger clusters to tolerate more contamination.
This yields L-ratio, which can be expressed as:

.. math::
    L_{\mathrm{ratio}}(C) = \frac{L(C)}{N_s}



Expectation and use
-------------------

Since this metric identifies unit separation, a high value indicates a highly contaminated unit (type I error)
([Schmitzer-Torbert]_ et al.). [Jackson]_ et al. suggests that this measure is also correlated with type II errors
(although more strongly with type I errors).

A well separated unit should have a low L-ratio ([Schmitzer-Torbert]_ et al.).


Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    _, l_ratio = sqm.isolation_distance(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)


References
----------

.. autofunction:: spikeinterface.qualitymetrics.pca_metrics.mahalanobis_metrics
    :noindex:

Literature
----------

Introduced by [Schmitzer-Torbert]_ et al..
Early discussion and comparison with isolation distance by [Jackson]_ et al..
