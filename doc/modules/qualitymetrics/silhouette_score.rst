Silhouette score
============================================

Calculation
-----------

Gives the ratio between the cohesiveness of a cluster and its separation from other clusters.
Values for silhouette score range from -1 to 1. First the pairwise distances between each
point and every other point :math:`a(i)` in a cluster :math:`C_i` and then iterating through 
every other cluster distances between the points in :math:`C_i` and the points in :math:`C_j` 
are calculated. The cluster with the minimal mean distance is taken to be :math:`b(i)`. The
average value of :math:`s(i)` is taken to give the final silhouette score for a cluster.

.. math::
    a(i) = \frac{1}{C_i-1} \sum_{j \in C_i, j \neq i} d(i,j)
    b(i) = \min {J \neq I} \frac{1}{C_j} \sum_{j \in C_j} d(i, j)
    s(i) = \frac{a(i)-b(i)}{\max(a(i), b(i))}
    Silhouette Score = \frac{1}{N} \sum^{N} s(i)

Expectation and use
-------------------

A good clustering with well separated and compact clusters will have a silhouette score close to 1.
A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error).


Literature
----------

Introduced by [Rousseeuw]_.
