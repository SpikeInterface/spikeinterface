Mahalanobis metrics (:code:`isolation_distance`, :code:`l_ratio`)
=================================================================

Mahalanobis metrics are quality metrics based on the Mahalanobis distance between spikes and cluster centres in the PCA space.

They include:
- Isolation distance (:code:`isolation_distance`)
- L-ratio (:code:`l_ratio`)

Calculation
-----------

Isolation distance
~~~~~~~~~~~~~~~~~~

- :math:`C` : cluster of interest.
- :math:`N_s` : number of spikes within cluster :math:`C`.
- :math:`N_n` : number of spikes outside of cluster :math:`C`.
- :math:`N_{min}` : minimum of :math:`N_s` and :math:`N_n`.
- :math:`\mu_C`, :math:`\Sigma_C` : mean vector and covariance matrix for spikes within :math:`C` (where each spike within :math:`C` is represented by a vector of principal components (PCs)).
- :math:`D_{i,C}^2` : for every spike :math:`i` (represented by vector :math:`x_i`) outside of cluster :math:`C`, the Mahalanobis distance (as below) between :math:`\mu_c` and :math:`x_i` is calculated. These distances are ordered from smallest to largest. The :math:`N_{min}`'th entry in this list is the isolation distance.

.. math::
    D_{i,C}^2 = (x_i - \mu_C)^T \Sigma_C^{-1} (x_i - \mu_C)

Geometrically, the isolation distance for cluster :math:`C` is the radius of the circle which contains :math:`N_{min}` spikes from cluster :math:`C` and :math:`N_{min}` spikes outside of the cluster :math:`C`.

L-ratio
~~~~~~~

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

Isolation distance can be interpreted as a measure of distance from the cluster to the nearest other cluster.
A well isolated unit should have a large isolation distance.

L-ratio quantifies unit separation, so a high value indicates a highly contaminated unit (type I error)
([Schmitzer-Torbert]_ et al.). [Jackson]_ et al. suggests that this measure is also correlated with type II errors
(although more strongly with type I errors).

A well separated unit should have a low L-ratio ([Schmitzer-Torbert]_ et al.).


Example code
------------

.. code-block:: python

    from spikeinterface.metrics.quality.pca_metrics import mahalanobis_metrics

    isolation_distance, l_ratio = mahalanobis_metrics(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)


References
----------

.. autofunction:: spikeinterface.metrics.quality.pca_metrics.mahalanobis_metrics
    :noindex:

Literature
----------

Isolation distance introduced by [Harris]_.
L-ratio introduced by [Schmitzer-Torbert]_ et al..
Early discussion and comparison with isolation distance by [Jackson]_ et al..
