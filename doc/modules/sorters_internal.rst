.. _internal_sorters:

Internal sorters
================

:py:mod:`spikeinterface.sortingcomponents` implement algorithms to break into components
a sorting pipeline. With this components it is easy to develop a new sorter.

Theses components and sorters havs been benchmarked [here](https://github.com/samuelgarcia/sorting_components_benchmark_paper).


At the moment 4 internal sorters are implemented in spikeinterface:

* :code:`lupin`
* :code:`spykingcircus2`
* :code:`tridesclous2`
* :code:`simple`

Please read more details below.


Lupin
-----

Lupin is components-based sorters, it combine components that give the best reults on benchmarks
for each steps. It is theorically the "best" sorter, spikeiterface can offer internally.

In summary, Lupin uses: 
  * preprocessing (filtering, CMR, whitening)
  * the *DREDGE* motion correction algorithm (optional)
  * peak detection with *matched filtering*
  * iterative splits for clustering *Iter-ISOPLIT*
  * augmented matching pursuit for the spike deconvolution with *Wobble*

Some note on this algos and related parameters:
  * waveforms size is different for clustering and template matching
    `clustering_ms_before` `clustering_ms_after` `ms_before` `ms_after`
  * the filtering is quite smooth `freq_max=7000.`
  * `n_pca_features` can impact the clustering step
  * there is a clean step before the template matching using `template_sparsify_threshold` 
    `template_min_snr_ptp` `template_max_jitter_ms` `min_firing_rate`. This can impact a lot the result.
  * Lupin is a bit slower than `tridesclous2`` and `spkykingcircus2`

SpyKING-CIRCUS 2
----------------

This is an updated version of SpyKING-CIRCUS [Yger2018]_ based on the modular
components. In summary, this spike sorting pipeline uses optionaly the DREDGE motion
correction algorithm before filtering and whitening the data. On these whitened data, the chains of components
that are used are: matched filtering for peak detection, iterative splits for clustering (Iter-HDBSCAN),
and orthogonal matching pursuit for template reconstruction (Circus-OMP).


TriDesClous 2
-------------

This is an updated version of TriDesClous based on the modular components.
This do not give as good results as `lupin` but this was faster.
This is sorter is a good choice for very fast exploration of a dataset.

Internally tridesclous2 uses: 
  * preprocessing (filtering, CMR) but no whitening
  * the *DREDGE* motion correction algorithm (optional)
  * peak detection with *locally_exlusive*
  * iterative splits for clustering *Iter-ISOPLIT*
  * fast template matching using the *TDC-peeler*


Simple
------

This is a simple sorter that **do not use the template matching**.
It can be seen as an "old school" sorters with only peak detection, features reduction (svd) and
clustering.
Using this sorter can be very very usefull on mono channel and tetrode dataset.
Very often on 1-4 channel dataset when the SNR is too small then template matching is an overkill
feature than give worst results.

The clustering step is quite flexible and several algos can be tested (kmeans, isosplit, hdbscan, ...)
