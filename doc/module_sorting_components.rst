Sorting Components module
=========================

Spike sorting is comprised of several steps, or components. In the :py:mod:`spikeinterface.sortingcomponents` module we
are building a library of methods and steps that can be assembled to build full spike sorting pipelines.

This effort goes in the direction of *modularization* of spike sorting algorithms. Currently, spike sorters are shipped
as full packages with all the steps needed to perform end-to-end spike sorting.

However, this might not be the best option. It is in fact very likely that a sorter has an excellent step,
say the clustering, but another step is sub-optimal. Decoupling different steps as separate components would allow
one to mix-and-match sorting steps from different sorters.

Another advantage of *modularization* is that we can accurately benchmark every step of a spike sorting pipeline.
For example, what is the performance of peak detection method 1 or 2, provided that the rest of the pipeline is the
same?

For now, we have methods for:
 * peak detection
 * peak localization
 * peak selection
 * motion estimation
 * motion correction
 * clustering
 * template matching

For some of theses steps, implementations are in avery early stage and are still a bit *drafty*.
Signature and behavior may change from time to time in this alpha period development.

You can also have a look `spikeinterface blog <https://spikeinterface.github.io>`_ where there are more detailled 
notebooks on sorting components.


Peak detection
--------------

Peak detection is usually the first step of spike sorting and it consists of finding peaks in the traces that could
be actual spikes.

Peaks can be detected with the :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks()` function as
follows:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    
    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)
    
    peaks = detect_peaks(
        recording, method='by_channel',
        peak_sign='neg',
        detect_threshold=5,
        n_shifts=2,
        local_radius_um=100,
        noise_levels=None,
        random_chunk_kwargs={},
        outputs='numpy_compact',
        **job_kwargs,
    )

The output :code:`peaks` is a numpy array with a length of the number of peaks found and the following dtype:

.. code-block:: python

    peak_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('amplitude', 'float64'), ('segment_ind', 'int64')]


Different methods are available with the :code:`method` argument:

* 'by_channel' (default): peaks are detected separately for each channel
* 'locally_exclusive': peaks on neighboring channels within a certain radius are excluded (not counted multiple times)

Peak detection, as many sorting components, can be run in parallel. Note that the 'locally_exclusive' method requires
:code:`numba` to be installed.


Peak localization
-----------------

Peak localization estimates the spike *location* on the probe. An estimate of location can be important to correct for
drifts or cluster spikes into different units.


Peak localization can be run using :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks()` as
follows:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    
    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)

    peak_locations = localize_peaks(recording, peaks, method='center_of_mass',
                                    local_radius_um=150, ms_before=0.3, ms_after=0.6,
                                    **job_kwargs)

                                        
Currently, the following methods are implemented:

  * 'center_of_mass' 
  * 'monopolar_triangulation' with optimizer='least_square'
    This method is from Julien Boussard and Erdem Varol from the Paninski lab.
    This has been presented at [NeurIPS](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26709)
    see also [here](https://openreview.net/forum?id=ohfi44BZPC4)
  * 'monopolar_triangulation' with optimizer='minimize_with_log_penality'

Theses methods are the same implemented in :py:mod:`spikeinterface.toolkit.postprocessing.unit_localization`



The output :code:`peak_locations` is a 1d numpy array with a dtype that depends on the chosen method.

For instance, the 'monopolar_triangulation' method will have:

.. code-block:: python

    localization_dtype = [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')]

.. note::

   By convention in spikeinterface, when a probe is describe in 2d
     * **'x'** is the width of the probe
     * **'y'** is the depth
     * **'z'** is the orthogonal to the probe plane


Peak selection
--------------

When too many peaks are detected a strategy can be used to select (or sub-sample) only some of them before clustering.
This is the strategy used by spyking-circus or tridesclous, for instance.
Then, clustering is run on this subset of peaks, templates are extracted, and a template-matching step is run to find 
all spikes.

The way the *peak vector* is reduced (or sub-sampled) is a crutial step because units with small firing rate
can be *hidden* by this process.


.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    
    many_peaks = detect_peaks(...)
    
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    
    some_peaks = select_peaks(many_peaks, method='uniform', n_peaks=10000)

Implemented methods are the following:

  * 'uniform'
  * 'uniform_locations'
  * 'smart_sampling_amplitudes'
  * 'smart_sampling_locations'
  * 'smart_sampling_locations_and_time'



Motion estimation
-----------------

Recently, drift estimation has been added in some the available spike sorters (Kilosort 2.5, 3)
Especially for Neuropixels-like probes, this is crucials step.

Several methods have been proposed to correct for drifts, but only one is currently implemented in spikeinterface 
at the moment. See `Decentralized Motion Inference and Registration of Neuropixel Data <https://ieeexplore.ieee.org/document/9414145>`_ 
for more details.

The motion estimation step comes after peak detection and peak localization.
The idea is to divide the recording in time bins and estimate the relative motion between temporal bins.

This method has two options:

  * rigid drift : one motion vector is estimated for the entire probe 
  * non-rigid drift : one motion vector is estimated per depth bin

Here is an example with non-rigid motion estimation
  
.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    peaks = detect_peaks(recording, ...)
    
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    peak_locations = localize_peaks(recording, peaks, ...)
    
    
    from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
    motion, temporal_bins, spatial_bins,
                extra_check = estimate_motion(recording, peaks, peak_locations=peak_locations,
                                              direction='y', bin_duration_s=1., bin_um=10., 
                                              margin_um=5,
                                              method='decentralized_registration', 
                                              method_kwargs={},
                                              non_rigid_kwargs={'bin_step_um': 50},
                                              output_extra_check=True,
                                              progress_bar=True, 
                                              verbose=True)    

In this example, because it is a non-rigid estimation, :code:`motion` is a 2d array (num_time_bins, num_spatial_bins).


Motion correction
-----------------

The estimated motion can be used to correct the motion, in other words, for drift correction.
One possible way is to make an interpolation sample-by-sample to compensate for the motion.
The :py:class:`~spikeinterface.sortingcomponents.motion_correction.CorrectMotionRecording` is a preprocessing step doing
this. This preprocessing is *lazy*, so that inperpolation is done the on-the-fly. However, the class needs the
"motion vector" as input, which requires a relatively long computation (peak detection, localization and motion
estimation).

Here is a short example the depends on the output of "Motion estimation":


.. code-block:: python

  from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
  
  recording_corrected = CorrectMotionRecording(recording_with_drift, motion, temporal_bins, spatial_bins)

**Important note**: currently, the borders of the probe in the y direction are NOT handled properly.
Therefore, it is safer to remove channel on the border after this step.
We plan to handle this directly in the class but this is NOT implemented yet.
Use this class carefully.

Clustering
----------

The clustering step remains the central step of the spike sorting.
Historically this step was separted into two distinct parts: feature reduction and clustering.
In spikeinterface, we decided to regroup this two steps in the same module.
This allows one to compute feature reduction on-the-fly and avoid long computations and storage of 
large features.

The clustering step takes the recording and detected (and optionally selected) peaks as input and returns 
a label for every peak.

At the moment, the implemenation is quite experimental.
These methods have been implemented:
  * "position_clustering": use HDBSCAN on peak locations.
  * "sliding_hdbscan": clustering approach from tridesclous, with sliding spatial windows. PCA and HDBSCAN are run 
    on local/sparse waveforms.
  * "position_pca_clustering": this method tries to use peak locations for a first clustering step and then perform 
  further splits using PCA + HDBSCAN

Different methods may need different inputs (for instance some of them require need peak locations and some do not).
    
.. code-block:: python
  
  from spikeinterface.sortingcomponents.peak_detection import detect_peaks
  peaks = detect_peaks(recording, ...)

  from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
  labels, peak_labels = find_cluster_from_peaks(recording, peaks, method="sliding_hdbscan")


* **labels** : contains all possible labels
* **peak_labels** : vector with the same size as peaks containing the label for each peak


Template matching
-----------------

Template matching is the final step used in many tools (kilosort, spyking-circus, yass, tridesclous, hdsort...)

In this step, from a given catalogue (or dictionnary) of templates (or atoms), the algorithms try to *explain* the 
traces as a linear sum of template plus a residual noise.

At the moment, there are four methods implemented:

  * 'naive': a very naive implemenation used as  a reference for benchmarks
  * 'tridesclous': the algorithm for template matching implemented in tridesclous
  * 'circus': the algorithm for template matching implemented in spyking-circus
  * 'circus-omp': a updated algorithm similar to the spyking-circus one circus but with OMP (orthogonal macthing 
    pursuit)

Very preliminary benchmarks suggest that:
 * 'circus-omp' is the most accurate, but a bit slow.
 * 'tridesclous' is the fastest and has very decent accuracy
