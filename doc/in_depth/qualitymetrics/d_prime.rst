D-prime (:code:`d_prime`)
=========================

Calculation
-----------

D-prime uses linear discriminant analysis (LDA) to estimate the classification accuracy between two units.

- :math:`v` denotes a waveform.
- :math:`C` denotes the class (unit) for which the metric is being calculated.
- :math:`D` denotes the set of spikes which are not in :math:`C`.
- :math:`P(v|C)` probability distributions are assumed to be Gaussian.

LDA is fit to spikes in :math:`C`, then to spikes in :math:`D`.

- :math:`\mu_C^{(LDA)}` and :math:`\mu_D^{(LDA)}` denote the mean of the LDA for clusters :math:`C` and :math:`D` respectively.
- :math:`\sigma_C^{(LDA)}` and :math:`\sigma_D^{(LDA)}` denote the standard deviation of the LDA for clusters :math:`C` and :math:`D` respectively.

D-prime is then calculated as follows:

.. math::
     D_{\mathrm{prime}}(C) = \frac{ ( \mu_C^{(LDA)} - \mu_D^{(LDA)} ) }{ \sqrt{ 0.5( (\sigma_C^{(LDA)})^2 + (\sigma_D^{(LDA)})^2) } }


Expectation and use
-------------------

D-prime is a measure of cluster separation, and will be larger in well separated clusters.

Example code
------------

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    d_prime = sqm.lda_metrics(all_pcs=all_pcs, all_labels=all_labels, this_unit_id=0)


Reference
---------

.. autofunction:: spikeinterface.qualitymetrics.pca_metrics.lda_metrics


Literature
----------

Introduced by [Hill]_.
