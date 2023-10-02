Quality Metrics module
======================

Quality metrics allows one to quantitatively assess the *goodness* of a spike sorting output.
The :py:mod:`~spikeinterface.qualitymetrics` module includes functions to compute a large variety of available metrics.
All of the metrics currently implemented in spikeInterface are *per unit* (pairwise metrics do appear in the literature).

Each metric aims to identify some quality of the unit.
Contamination metrics (also sometimes called 'false positive' or 'type I' metrics) aim to identify the amount of noise present in the unit.
Examples include: ISI violations, sliding refractory period violations, SNR, NN-hit rate.
Completeness metrics (or 'false negative'/'type II' metrics) aim to identify whether any of the spiking activity is missing from a unit.
Examples include: presence ratio, amplitude cutoff, NN-miss rate.
Drift metrics aim to identify changes in waveforms which occur when spike sorters fail to successfully track neurons in the case of electrode drift.

Some metrics make use of principal component analysis (PCA) to reduce the dimensionality of computations.
Various approaches to computing the principal components are possible, and choice should be carefully considered in relation to the recording equipment used.
The following metrics make use of PCA: isolation distance, L-ratio, D-prime, Silhouette score and NN-metrics.
By contrast, the following metrics are based on spike times only: firing rate, ISI violations, presence ratio.
And amplitude cutoff and SNR are based on spike times as well as waveforms.

For more details about each metric and it's availability and use within SpikeInterface, see the individual pages for each metrics.

.. toctree::
  :maxdepth: 1
  :glob:

  qualitymetrics/amplitude_cutoff
  qualitymetrics/amplitude_cv
  qualitymetrics/amplitude_median
  qualitymetrics/d_prime
  qualitymetrics/drift
  qualitymetrics/firing_range
  qualitymetrics/firing_rate
  qualitymetrics/isi_violations
  qualitymetrics/isolation_distance
  qualitymetrics/l_ratio
  qualitymetrics/nearest_neighbor
  qualitymetrics/presence_ratio
  qualitymetrics/sliding_rp_violations
  qualitymetrics/snr
  qualitymetrics/noise_cutoff
  qualitymetrics/silhouette_score
  qualitymetrics/synchrony


This code snippet shows how to compute quality metrics (with or without principal components) in SpikeInterface:

.. code-block:: python

    we = si.load_waveforms(folder='waveforms') # start from a waveform extractor

    # without PC
    metrics = compute_quality_metrics(waveform_extractor=we, metric_names=['snr'])
    assert 'snr' in metrics.columns

    # with PCs
    from spikeinterface.postprocessing import compute_principal_components
    pca = compute_principal_components(waveform_extractor=we, n_components=5, mode='by_channel_local')
    metrics = compute_quality_metrics(waveform_extractor=we)
    assert 'isolation_distance' in metrics.columns

For more information about quality metrics, check out this excellent
`documentation <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html>`_
from the Allen Institute.


Quality Metrics References
--------------------------

.. toctree::
  :maxdepth: 1

  qualitymetrics/references
