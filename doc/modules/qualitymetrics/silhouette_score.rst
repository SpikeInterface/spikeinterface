Silhouette score (not currently implemented)
============================================

Calculation
-----------

Gives the ratio between the cohesiveness of a cluster and its separation from other clusters.
Values for silhouette score range from -1 to 1.

Expectation and use
-------------------

A good clustering with well separated and compact clusters will have a silhouette score close to 1.
A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error).


Literature
----------

Introduced by [Rousseeuw]_.
