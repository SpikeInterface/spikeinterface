Isolation distance (:code:`isolation_distance`)
===============================================

Calculation
-----------

- :math:`C` : cluster of interest.
- :math:`N_s` : number of spikes within cluster :math:`C`.
- :math:`N_n` : number of spikes outside of cluster :math:`C`.
- :math:`N_{min}` : minimum of :math:`N_s` and :math:`N_n`.
- :math:`\mu_C`, :math:`\Sigma_C` : mean vector and covariance matrix for spikes within :math:`C` (where each spike within :math:`C` is represented by a vector of principal components (PCs)).
- :math:`D_{i,C}^2` : for every spike :math:`i` (represented by vector :math:`x_i`) outside of cluster :math:`C`, the Mahalanobis distance (as below) between :math:`\mu_c` and :math:`x_i` is calculated. These distances are ordered from smallest to largest. The :math:`N_{min}`'th entry in this list is the isolation distance.

.. math::
    D_{i,C}^2 = (x_i - \mu_C)^T \Sigma_C^{-1} (x_i - \mu_C)

Geometrically, the isolation distance for cluster :math:`C` is the radius of the circle which contains :math:`N_{min}` spikes from cluster :math:`C` and :math:`N_{min}` spikes outside of the cluster :math:`C`.


Expectation and use
-------------------

Isolation distance can be interpreted as a measure of distance from the cluster to the nearest other cluster.
A well isolated unit should have a large isolation distance.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    iso_distance, _ = sqm.isolation_distance(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)


References
----------

.. autofunction:: spikeinterface.qualitymetrics.pca_metrics.mahalanobis_metrics


Literature
----------

Introduced by [Harris]_.
