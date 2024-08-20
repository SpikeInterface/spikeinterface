How to Cite
===========

If you like SpikeInterface, please star us on `Github <https://github.com/SpikeInterface/spikeinterface>`_!
*giving us a star gives a measure of the level of use and interest, which goes a long way to getting funding*

Please cite SpikeInterface in your papers with our eLife paper: [Buccino]_

SpikeInterface stands on the shoulders of giants!
Each method in SpikeInterface draws on (or directly runs) independently-created methods.
Please try to reference the individual works that are important for your analysis pipeline.
If you notice a missing reference, please let us know by `submitting an issue <https://github.com/SpikeInterface/spikeinterface/issues/new>`_ on Github.

Preprocessing Module
--------------------
If you use one of the following preprocessing methods, please cite the appropriate source:

- :code:`phase_shift` or :code:`highpass_spatial_filter` [IBL]_
- :code:`detect_bad_channels(method='coherence+psd')` [IBL]_
- :code:`common_reference` [Rolston]_

Motion Correction
^^^^^^^^^^^^^^^^^
If you use the :code:`correct_motion` method in the preprocessing module, please cite [Garcia]_
as well as the references that correspond to the :code:`preset` you used:

- :code:`nonrigid_accurate` [Windolf]_ [Varol]_
- :code:`nonrigid_fast_and_accurate` [Windolf]_ [Varol]_ [Pachitariu]_
- :code:`rigid_fast` *no additional citation needed*
- :code:`kilosort_like` [Pachitariu]_

Sorters Module
--------------
If you use one of the following spike sorting algorithms (i.e. you use the :code:`run_sorter()` method,
please include the appropriate citation for the :code:`sorter_name` parameter you use:
*Note: unless otherwise stated, the reference given is to be used for all versions of the sorter*

- :code:`combinato` [Niediek]_
- :code:`hdsort` [Diggelmann]_
- :code:`herdingspikes` [Muthmann]_ [Hilgen]_
- :code:`kilosort`  [Pachitariu]_
- :code:`mountainsort` [Chung]_
- :code:`spykingcircus` [Yger]_
- :code:`wavclus` [Chaure]_
- :code:`yass` [Lee]_

Qualitymetrics Module
---------------------
If you use the :code:`qualitymetrics` module, i.e. you use the :code:`analyzer.compute()`
or :code:`compute_quality_metrics()` methods, please include the citations for the :code:`metric_names` that were particularly
important for your research:

- :code:`amplitude_cutoff` [Hill]_
- :code:`amplitude_median` [IBL]_
- :code:`sliding_rp_violation` [IBL]_
- :code:`drift` [Siegle]_
- :code:`isi_violation` [UMS]_
- :code:`rp_violation` [Llobet]_
- :code:`sd_ratio` [Pouzat]_
- :code:`snr` [Lemon]_ [Jackson]_
- :code:`synchrony` [Grün]_

If you use the :code:`qualitymetrics.pca_metrics` module, i.e. you use the
:code:`compute_pc_metrics()` method, please include the citations for the :code:`metric_names` that were particularly
important for your research:

- :code:`d_prime` [Hill]_
- :code:`isolation_distance` or :code:`l_ratio` [Schmitzer-Torbert]_
- :code:`nearest_neighbor` or :code:`nn_isolation` or :code:`nn_noise_overlap` [Chung]_ [Siegle]_
- :code:`silhouette`  [Rousseeuw]_ [Hruschka]_

Curation Module
---------------
If you use the :code:`get_potential_auto_merge` method from the curation module, please cite [Llobet]_

References
----------

.. [Buccino] `SpikeInterface, a unified framework for spike sorting. 2020. <https://pubmed.ncbi.nlm.nih.gov/33170122/>`_

.. [Buzsáki] `The Log-Dynamic Brain: How Skewed Distributions Affect Network Operations. 2014. <https://pubmed.ncbi.nlm.nih.gov/24569488/>`_

.. [Chaure] `A novel and fully automatic spike-sorting implementation with variable number of features. 2018. <https://pubmed.ncbi.nlm.nih.gov/29995603/>`_

.. [Chung] `A Fully Automated Approach to Spike Sorting. 2017. <https://pubmed.ncbi.nlm.nih.gov/28910621/>`_

.. [Diggelmann] `Automatic spike sorting for high-density microelectrode arrays. 2018. <https://pubmed.ncbi.nlm.nih.gov/30207864/>`_

.. [Garcia] `A Modular Implementation to Handle and Benchmark Drift Correction for High-Density Extracellular Recordings. 2024. <https://pubmed.ncbi.nlm.nih.gov/38238082/>`_

.. [Grün] `Impact of higher-order correlations on coincidence distributions of massively parallel data. 2007. <https://www.researchgate.net/publication/225145104_Impact_of_Higher-Order_Correlations_on_Coincidence_Distributions_of_Massively_Parallel_Data>`_

.. [Harris] `Temporal interaction between single spikes and complex spike bursts in hippocampal pyramidal cells. 2001. <https://pubmed.ncbi.nlm.nih.gov/11604145/>`_

.. [Hilgen] `Unsupervised Spike Sorting for Large-Scale, High-Density Multielectrode Arrays. 2017. <https://pubmed.ncbi.nlm.nih.gov/28273464/>`_

.. [Hill] `Quality Metrics to Accompany Spike Sorting of Extracellular Signals. 2011. <https://pubmed.ncbi.nlm.nih.gov/21677152/>`_

.. [Hruschka] `Evolutionary algorithms for clustering gene-expression data. 2004. <https://www.researchgate.net/publication/220765683_Evolutionary_Algorithms_for_Clustering_Gene-Expression_Data>`_

.. [IBL] `Spike sorting pipeline for the International Brain Laboratory. 2022. <https://figshare.com/articles/online_resource/Spike_sorting_pipeline_for_the_International_Brain_Laboratory/19705522/3>`_

.. [Jackson] Quantitative assessment of extracellular multichannel recording quality using measures of cluster separation. Society of Neuroscience Abstract. 2005.

.. [Lee] `YASS: Yet another spike sorter. 2017. <https://www.biorxiv.org/content/10.1101/151928v1>`_

.. [Lemon] Methods for neuronal recording in conscious animals. IBRO Handbook Series. 1984.

.. [Llobet] `Automatic post-processing and merging of multiple spike-sorting analyses with Lussac. 2022. <https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1>`_

.. [Muthmann] `Spike Detection for Large Neural Populations Using High Density Multielectrode Arrays. 2015. <https://pubmed.ncbi.nlm.nih.gov/26733859/>`_

.. [Niediek] `Reliable Analysis of Single-Unit Recordings from the Human Brain under Noisy Conditions: Tracking Neurons over Hours. 2016. <https://pubmed.ncbi.nlm.nih.gov/27930664/>`_

.. [Pachitariu] `Spike sorting with Kilosort4. 2024. <https://pubmed.ncbi.nlm.nih.gov/38589517/>`_

.. [Pouzat] `Using noise signature to optimize spike-sorting and to assess neuronal classification quality. 2002. <https://pubmed.ncbi.nlm.nih.gov/12535763/>`_

.. [Rolston] `Common median referencing for improved action potential detection with multielectrode arrays. 2009. <https://pubmed.ncbi.nlm.nih.gov/19964004/>`_

.. [Rousseeuw] `Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. 1987. <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

.. [Schmitzer-Torbert] `Neuronal Activity in the Rodent Dorsal Striatum in Sequential Navigation: Separation of Spatial and Reward Responses on the Multiple T Task. 2004. <https://pubmed.ncbi.nlm.nih.gov/14736863/>`_

.. [Siegle] `Survey of Spiking in the Mouse Visual System Reveals Functional Hierarchy. 2021. <https://pubmed.ncbi.nlm.nih.gov/33473216/>`_

.. [UMS] `UltraMegaSort2000 - Spike sorting and quality metrics for extracellular spike data. 2011. <https://github.com/danamics/UMS2K>`_

.. [Varol] `Decentralized Motion Inference and Registration of Neuropixel Data. 2021. <https://ieeexplore.ieee.org/document/9414145>`_

.. [Windolf] `Robust Online Multiband Drift Estimation in Electrophysiology Data. 2022. <https://www.biorxiv.org/content/10.1101/2022.12.04.519043v2>`_

.. [Yger] `A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. 2018. <https://pubmed.ncbi.nlm.nih.gov/29557782/>`_
