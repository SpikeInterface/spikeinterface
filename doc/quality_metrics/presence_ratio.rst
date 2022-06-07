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

Literature
----------

Unknown origin.

Citations
---------

