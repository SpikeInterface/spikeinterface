How to Cite
==========

If you use Spikeinterface, please star us on Github!
Please cite Spikeinterface in your papers with the following reference:


Spikeinterface stands on the shoulders of giants!
Each method in Spikeinterface draws on (or directly runs) methods made by dozens of individuals.
Please try to reference the individual works that are important for your analysis pipeline.

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

- :code:`amplitude_cutoff` :code:`isi_violation` [Hill]_
- :code:`amplitude_median` [IBL]_
- :code:`drift` [Siegle]_
- :code:`rp_violation` [Llobet]_
- :code:`sd_ratio` [Pouzat]_
- :code:`sliding_rp_violation` [IBL]_
- :code:`snr` [Lemon]_ [Jackson]_
- :code:`synchrony` [Gruen]_

If you use the :code:`qualitymetrics.pca_metrics` module, i.e. you use the
:code:`calculate_pc_metrics()` method, please include the citations for the :code:`metric_names` that were particularly
important for your research:

- :code:`d_prime` [Hill]_
- :code:`isolation_distance` :code:`l_ratio` [Schmitzer-Torbert]_
- :code:`nearest_neighbor` :code:`nn_isolation` :code:`nn_noise_overlap` [Chung]_  [Siegle]_
- :code:`silhouette` [Hill]_




Nearest Neighbor Metrics (nn_hit_rate, nn_miss_rate, nn_isolation, nn_noise_overlap)
Silhouette score (silhouette, silhouette_full)


list of metric_names for quality_metrics:
"num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rates,
    "presence_ratio": compute_presence_ratios,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "rp_violation": compute_refrac_period_violations,
    "sliding_rp_violation": compute_sliding_rp_violations,
    "amplitude_cutoff": compute_amplitude_cutoffs,
    "amplitude_median": compute_amplitude_medians,
    "amplitude_cv": compute_amplitude_cv_metrics,
    "synchrony": compute_synchrony_metrics,
    "firing_range": compute_firing_ranges,
    "drift": compute_drift_metrics,
    "sd_ratio": compute_sd_ratio,

list of metric_names for pc_metrics:
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",




References
----------
.. [Niediek] Niediek J, Boström J, Elger CE, Mormann F. Reliable Analysis of Single-Unit Recordings from the Human Brain under Noisy Conditions: Tracking Neurons over Hours. PLoS One. 2016 Dec 8;11(12):e0166598. doi: 10.1371/journal.pone.0166598. PMID: 27930664; PMCID: PMC5145161.

.. [Diggelmann] Diggelmann R, Fiscella M, Hierlemann A, Franke F. Automatic spike sorting for high-density microelectrode arrays. J Neurophysiol. 2018 Dec 1;120(6):3155-3171. doi: 10.1152/jn.00803.2017. Epub 2018 Sep 12. PMID: 30207864; PMCID: PMC6314465.

.. [Muthmann] Muthmann JO, Amin H, Sernagor E, Maccione A, Panas D, Berdondini L, Bhalla US, Hennig MH. Spike Detection for Large Neural Populations Using High Density Multielectrode Arrays. Front Neuroinform. 2015 Dec 18;9:28. doi: 10.3389/fninf.2015.00028. PMID: 26733859; PMCID: PMC4683190.

.. [Hilgen] Hilgen G, Sorbaro M, Pirmoradian S, Muthmann JO, Kepiro IE, Ullo S, Ramirez CJ, Puente Encinas A, Maccione A, Berdondini L, Murino V, Sona D, Cella Zanacchi F, Sernagor E, Hennig MH. Unsupervised Spike Sorting for Large-Scale, High-Density Multielectrode Arrays. Cell Rep. 2017 Mar 7;18(10):2521-2532. doi: 10.1016/j.celrep.2017.02.038. PMID: 28273464.

.. [Pachitariu] Pachitariu M, Sridhar S, Pennington J, Stringer C. Spike sorting with Kilosort4. Nat Methods. 2024 May;21(5):914-921. doi: 10.1038/s41592-024-02232-7. Epub 2024 Apr 8. PMID: 38589517; PMCID: PMC11093732.

.. [Chung] Chung JE, Magland JF, Barnett AH, Tolosa VM, Tooker AC, Lee KY, Shah KG, Felix SH, Frank LM, Greengard LF. A Fully Automated Approach to Spike Sorting. Neuron. 2017 Sep 13;95(6):1381-1394.e6. doi: 10.1016/j.neuron.2017.08.030. PMID: 28910621; PMCID: PMC5743236.

.. [Yger] Yger P, Spampinato GL, Esposito E, Lefebvre B, Deny S, Gardella C, Stimberg M, Jetter F, Zeck G, Picaud S, Duebel J, Marre O. A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo. Elife. 2018 Mar 20;7:e34518. doi: 10.7554/eLife.34518. PMID: 29557782; PMCID: PMC5897014.

.. [Chaure] Chaure FJ, Rey HG, Quian Quiroga R. A novel and fully automatic spike-sorting implementation with variable number of features. J Neurophysiol. 2018 Oct 1;120(4):1859-1871. doi: 10.1152/jn.00339.2018. Epub 2018 Jul 11. PMID: 29995603; PMCID: PMC6230803.

.. [Lee] Lee JH, Carlson D, Shokri H, Yao W, Goetz G, Hagen E, Batty E, Chichilnisky EJ, Einevoll G, Paninski L. YASS: Yet another spike sorter. bioRxiv 151928; doi: https://doi.org/10.1101/151928 . Epub 2017

.. [Buzsáki] Buzsáki, György, and Kenji Mizuseki. “The Log-Dynamic Brain: How Skewed Distributions Affect Network Operations.” Nature reviews. Neuroscience 15.4 (2014): 264–278. Web.

.. [Chung] Chung, Jason E et al. “A Fully Automated Approach to Spike Sorting.” Neuron (Cambridge, Mass.) 95.6 (2017): 1381–1394.e6. Web.

.. [Harris] Kenneth D Harris, Hajime Hirase, Xavier Leinekugel, Darrell A Henze, and Gy ̈orgy Buzs ́aki. Temporal interaction between single spikes and complex spike bursts in hippocampal pyramidal cells. Neuron (Cambridge, Mass.), 32(1):141–149, 2001.

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.

.. [Hruschka] Hruschka, E.R., de Castro, L.N., Campello R.J.G.B. "Evolutionary algorithms for clustering gene-expression data." Fourth IEEE International Conference on Data Mining (ICDM'04) 2004, pp 403-406.

.. [Gruen] Sonja Grün, Moshe Abeles, and Markus Diesmann. Impact of higher-order correlations on coincidence distributions of massively parallel data. In International School on Neural Networks, Initiated by IIASS and EMFCSC, volume 5286, 96–114. Springer, 2007.

.. [IBL] International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022.

.. [Jackson] Jadin Jackson, Neil Schmitzer-Torbert, K.D. Harris, and A.D. Redish. Quantitative assessment of extracellular multichannel recording quality using measures of cluster separation. Soc Neurosci Abstr, 518, 01 2005.

.. [Lemon] R. Lemon. Methods for neuronal recording in conscious animals. IBRO Handbook Series, 4:56–60, 1984.

.. [Llobet] Llobet Victor, Wyngaard Aurélien and Barbour Boris. “Automatic post-processing and merging of multiple spike-sorting analyses with Lussac“. BioRxiv (2022).

.. [Pouzat] Pouzat Christophe, Mazor Ofer and Laurent Gilles. “Using noise signature to optimize spike-sorting and to assess neuronal classification quality“. Journal of Neuroscience Methods (2002).

.. [Rousseeuw] Peter J Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20(C):53–65, 1987.

.. [Schmitzer-Torbert]  Schmitzer-Torbert, Neil, and A. David Redish. “Neuronal Activity in the Rodent Dorsal Striatum in Sequential Navigation: Separation of Spatial and Reward Responses on the Multiple T Task.” Journal of neurophysiology 91.5 (2004): 2259–2272. Web.

.. [Siegle] Siegle, Joshua H. et al. “Survey of Spiking in the Mouse Visual System Reveals Functional Hierarchy.” Nature (London) 592.7852 (2021): 86–. Web.
