Quality Metrics module
======================

Quality metrics allows one to quantitatively assess the *goodness* of a spike sorting output.
The :py:mod:`~spikeinterface.metrics.quality` module includes functions to compute a large variety of available metrics.
All of the metrics currently implemented in spikeInterface are *per unit* (pairwise metrics do appear in the literature).

Each metric aims to identify some quality of the unit.
Contamination metrics (also sometimes called 'false positive' or 'type I' metrics) aim to identify the amount of noise present in the unit.
Examples include: ISI violations, sliding refractory period violations, SNR, NN-hit rate.
Completeness metrics (or 'false negative'/'type II' metrics) aim to identify whether any of the spiking activity is missing from a unit.
Examples include: presence ratio, amplitude cutoff, NN-miss rate.
Drift metrics aim to identify changes in waveforms which occur when spike sorters fail to successfully track neurons in the case of electrode drift.

The quality metrics are saved as an extension of a :doc:`SortingAnalyzer <../postprocessing>`. Some metrics can only be computed if certain extensions have been computed first. For example the drift metrics can only be computed the spike locations extension has been computed. By default, as many metrics as possible are computed. Which ones are computed depends on which other extensions have
been computed.

In detail, the default metrics are (click on each metric to find out more about them!):

- :doc:`qualitymetrics/firing_rate`
- :doc:`qualitymetrics/presence_ratio`
- :doc:`qualitymetrics/isi_violations`
- :doc:`qualitymetrics/sliding_rp_violations`
- :doc:`qualitymetrics/synchrony`
- :doc:`qualitymetrics/firing_range`

If :ref:`postprocessing_spike_locations` are computed, add:

- :doc:`qualitymetrics/drift`

If :ref:`postprocessing_spike_amplitudes` and ``templates`` are computed, add:

- :doc:`qualitymetrics/amplitude_cutoff`
- :doc:`qualitymetrics/amplitude_median`
- :doc:`qualitymetrics/amplitude_cv`
- :doc:`qualitymetrics/noise_cutoff`

If :ref:`postprocessing_noise_levels` and ``templates`` are computed, add:

- :doc:`qualitymetrics/snr`

If the recording, :ref:`postprocessing_spike_amplitudes` and ``templates`` are available, add:

- :doc:`qualitymetrics/sd_ratio`

If :ref:`postprocessing_principal_components` are computed, add:

- :doc:`qualitymetrics/mahalanobis`
- :doc:`qualitymetrics/d_prime`
- :doc:`qualitymetrics/silhouette_score`
- :doc:`qualitymetrics/nearest_neighbor` (note: excluding the ``nn_noise_overlap`` metric)

You can compute the default metrics using the following code snippet:

.. code-block:: python

    # load or create a sorting analyzer
    sorting_analyzer = si.load_sorting_analyzer(folder='my_sorting_analyzer')

    # compute the metrics
    sorting_analyzer.compute("quality_metrics")

    # get the metrics in the form of a pandas DataFrame
    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()

    # print the metrics that have been computed
    print(quality_metrics.columns)

Some metrics are very slow to compute when the number of units it large. So by default, the following metrics are not computed:

- The ``nn_advanced`` from :doc:`qualitymetrics/nearest_neighbor`

Some metrics make use of :ref:`principal component analysis <postprocessing_principal_components>` (PCA) to reduce the dimensionality of computations.
Various approaches to computing the principal components are possible, and choice should be carefully considered in relation to the recording equipment used.

If you only want to compute a subset of metrics, you can use convenience functions to compute each one,

.. code-block:: python

    from spikeinterface.quality_metrics import compute_isi_violations
    isi_violations = compute_isi_violations(sorting_analyzer, isi_threshold_ms=3.0)

This function returns the result of the computation but does not save it into the `sorting_analyzer`.
To save the result in your analyzer, you can use the ``compute`` method:

.. code-block:: python

    sorting_analyzer.compute(
        "quality_metrics",
        metric_names = ["isi_violation", "snr"],
        extension_params = {
            "isi_violation": {"isi_threshold_ms": 3.0},
        }
    )

Note that if you request a specific metric using ``metric_names`` and you do not have the required extension computed, the metric will be skipped.

For more information about quality metrics, check out this excellent
`documentation <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html>`_
from the Allen Institute.


.. toctree::
  :maxdepth: 1
  :glob:
  :hidden:

  qualitymetrics/amplitude_cutoff
  qualitymetrics/amplitude_cv
  qualitymetrics/amplitude_median
  qualitymetrics/d_prime
  qualitymetrics/drift
  qualitymetrics/firing_range
  qualitymetrics/firing_rate
  qualitymetrics/isi_violations
  qualitymetrics/mahalanobis
  qualitymetrics/nearest_neighbor
  qualitymetrics/noise_cutoff
  qualitymetrics/presence_ratio
  qualitymetrics/sd_ratio
  qualitymetrics/silhouette_score
  qualitymetrics/sliding_rp_violations
  qualitymetrics/snr
  qualitymetrics/synchrony
