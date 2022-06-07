Signal-to-noise (SNR) ratio
===========================

Calculation
-----------

- :math:`A_{\mu s}` : maximum amplitude of the mean spike waverform (possibly???).
- :math:`\sigma_b` : standard deviation of the background noise on one channel (noise).

.. math::
    \textrm{SNR} = \frac{A_{\mu s}}{\sigma_b}



Expectation and use
-------------------

A high SNR unit has a signal which is greater in amplitude than the background noise and is likely to correspond to a neuron [Jackson]_, [Lemon]_.
A low SNR value (close to 0) suggests that the unit is highly contaminated by noise (type I error).

References
----------

.. automodule:: spikeinterface.toolkit.qualitymetrics.misc_metrics

    .. autofunction:: compute_snrs

Literature
----------

Presented by Lemon_ and useful initial discussion by Jackson_.

Citations
---------

.. [Jackson] Jadin Jackson, Neil Schmitzer-Torbert, K.D. Harris, and A.D. Redish. Quanti-
tative assessment of extracellular multichannel recording quality using measures
of cluster separation. Soc Neurosci Abstr, 518, 01 2005

.. [Lemon] R. Lemon. Methods for neuronal recording in conscious animals. IBRO Hand-
book Series, 4:56–60, 1984