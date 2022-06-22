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

	import spikeinterface.toolkit as st

    d_prime = st.lda_metrics(all_pcs, all_labels, 0)

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/ccaec9bac37b0b7d31c955594780c706fe165c37/spikeinterface/toolkit/qualitymetrics/pca_metrics.py#L188>`_

Reference
---------

.. automodule:: spikeinterface.toolkit.qualitymetrics.pca_metrics

    .. autofunction:: lda_metrics

Used to measure cluster separation.
The magnitude of D-prime will be higher in well separated clusters, and is therefore expected to be higher in true positive units.


Literature
----------

Introduced by [Hill]_.

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.
