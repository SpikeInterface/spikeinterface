.. _internal_sorters:

Internal sorters
================

:py:mod:`spikeinterface.sortingcomponents` implement algorithms to break a sorting pipeline
into individual components. With this components it is easy to develop a new sorter.

These components and sorters havs been benchmarked [here](https://github.com/samuelgarcia/sorting_components_benchmark_paper).


At the moment, there are 4 internal sorters implemented in ``spikeinterface``:

* :code:`lupin`
* :code:`spykingcircus2`
* :code:`tridesclous2`
* :code:`simple`


Lupin
-----

Lupin is components-based sorters, it combine components that give the best reults on benchmarks
for each steps. It is theorically the "best" sorter that ``spikeinterface`` can offer internally.

Lupin components are:
  * preprocessing (filtering, CMR, whitening)
  * the *DREDGE* motion correction algorithm (optional)
  * peak detection with *matched filtering*
  * iterative splits for clustering *Iter-ISOPLIT*
  * augmented matching pursuit for the spike deconvolution with *Wobble*


Some notes on this algorithm and related parameters:
  * waveforms size is different for clustering and template matching:
    ``clustering_ms_before``, ``clustering_ms_after``, ``ms_before``, ``ms_after``
  * the filtering is quite smooth by default to filter out high-frequency noise: ``freq_max=7000``
  * ``n_pca_features`` can impact the clustering step
  * there is a cleaning step before the template matching using ``template_sparsify_threshold``,
    ``template_min_snr_ptp``, ``template_max_jitter_ms``, and ``min_firing_rate``. This step can have a substantial impact on the result.
  * Lupin is a bit slower than ``tridesclous2`` and ``spkykingcircus2``, but more accurate!

SpyKING-CIRCUS 2
----------------

This is an updated version of SpyKING-CIRCUS [Yger2018]_ based on the modular
components. In summary, this spike sorting pipeline uses optionaly the DREDGE motion
correction algorithm before filtering and whitening the data. On these whitened data, the chains of components
that are used are: matched filtering for peak detection, iterative splits for clustering (Iter-HDBSCAN),
and orthogonal matching pursuit for template reconstruction (Circus-OMP).

SpyKING-CIRCUS 2 components are:
  * preprocessing (filtering, CMR, whitening)
  * the *DREDGE* motion correction algorithm (optional)
  * peak detection with *matched filtering*
  * iterative splits for clustering *Iter-HDBSCAN*
  * orthogonal matching pursuit for the spike deconvolution with *Circus-OMP*

TriDesClous 2
-------------

This is an updated version of TriDesClous based on the modular components.
It is not as good as ``Lupin`` in terms of performance, but it's way faster.
This is sorter is a good choice for a very fast exploration of a dataset.

TriDesClous 2 components are:
  * preprocessing (filtering, CMR) but no whitening
  * the *DREDGE* motion correction algorithm (optional)
  * peak detection with *locally_exlusive*
  * iterative splits for clustering *Iter-ISOPLIT*
  * fast template matching using the *TDC-peeler*


Simple
------

This is a simple sorter that **does not use the template matching**.
It can be seen as an "old school" sorter with only peak detection, feature reduction (svd) and
clustering.
Using this sorter can be very useful on mono channel and tetrode datasets.
Very often on 1-4 channel dataset when the SNR is too small then template matching is an overkill
feature that gives worse results.

The clustering step is quite flexible and several algorithms can be tested (k-means, isosplit, hdbscan, ...)
