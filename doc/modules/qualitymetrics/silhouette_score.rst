.. _silhouette_score :

Silhouette score (:code:`silhouette`, :code:`silhouette_full`)
==============================================================

Calculation
-----------

Gives the ratio between the cohesiveness of a cluster and its separation from other clusters.
Values for silhouette score range from -1 to 1.

For the full method as proposed by [Rousseeuw]_, the pairwise distances between each point
and every other point :math:`a(i)` in a cluster :math:`C_i` are calculated and then iterating through
every other cluster's distances between the points in :math:`C_i` and the points in :math:`C_j`
are calculated. The cluster with the minimal mean distance is taken to be :math:`b(i)`. The
average value of :math:`s(i)` is taken to give the final silhouette score for a cluster with
the following equations:

.. math::
    a(i) = \frac{1}{C_i-1} \sum_{j \in C_i, j \neq i} d(i,j)

    b(i) = \min {J \neq I} \frac{1}{C_j} \sum_{j \in C_j} d(i, j)

    s(i) = \frac{a(i)-b(i)}{\max(a(i), b(i))}

    Silhouette Score = \frac{1}{N} \sum^{N} s(i)

In order to improve computational complexity an alternative approach was proposed by [Hruschka]_
in which rather than pairwise point calculations, centroids of clusters are used. Thus :math:`a(i)`
is determined by distances from each point :math:`i` to the centroid of :math:`C_I` given as
:math:`mu_{C_I}`, which means the calculations are simplified as follows:

.. math::
    a(i) = d(i, \mu_{C_I})

    b(i) = \min {C_J \neq C_I}  d(i, \mu_{C_J})

    s(i) = \frac{a(i)-b(i)}{\max(a(i), b(i))}

    Silhouette Score = \frac{1}{N} \sum^{N} s(i)

Expectation and use
-------------------

A good clustering with well separated and compact clusters will have a silhouette score close to 1.
A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error).
SpikeInterface provides access to both implementations of silhouette score.

To reduce complexity the default implementation in SpikeInterface is to use the simplified silhouette score.
This can be changes by switching the silhouette method to either 'full' (the Rousseeuw implementation) or
('simplified', 'full') for both methods when entering the qm_params parameter.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    simple_sil_score = sqm.simplified_silhouette_score(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)


References
----------

.. autofunction:: spikeinterface.qualitymetrics.pca_metrics.simplified_silhouette_score

.. autofunction:: spikeinterface.qualitymetrics.pca_metrics.silhouette_score


Literature
----------

Full method introduced by [Rousseeuw]_.
Simplified method introduced by [Hruschka]_.
