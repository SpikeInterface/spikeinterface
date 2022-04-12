Sorting Components module
=========================

Spike sorting is comprised of several steps, or components. In the :code:`sortingcomponents` module we are building a
library of methods and steps that can be assembled to build full spike sorting pipelines.

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

For some of theses steps, implementations are in early stage and are still a bit drafty.
Signature and behavior may change from time to time.

You can also have a look `spikeinterface blog <https://spikeinterface.github.io>`_ where have have more detailled notebook
on sorting components.


Peak detection
--------------

Peak detection is usually the first step of spike sorting and it consists of finding peaks in the traces that could
be actual spikes.

Peaks can be detected with the :code:`detect_peaks()` function as follows:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    
    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)
    
    peaks = detect_peaks(recording, method='by_channel',
                             peak_sign='neg', detect_threshold=5, n_shifts=2,
                             local_radius_um=100,
                             noise_levels=None,
                             random_chunk_kwargs={},
                             outputs='numpy_compact',
                             **job_kwargs)

The output :code:`peaks` is a numpy array with a length of the number of peaks found and the following dtype:

.. code-block:: python

    peak_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('amplitude', 'float64'), ('segment_ind', 'int64')]


Different methods are available with the :code:`method` argument:

* 'by_channel' (default): peaks are detected separately for each channel
* 'locally_exclusive': peaks on neighboring channels within a certain radius are excluded (not counted multiple times)

Peak detection, as many sorting components, can be run in parallel.


Peak localization
-----------------

Peak localization estimates the spike *location* on the probe. An estimate of location can be important to correct for
drifts or cluster spikes into different units.



Peak localization can be run as follows:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    
    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)

    peak_locations = localize_peaks(recording, peaks, method='center_of_mass',
                                        local_radius_um=150, ms_before=0.3, ms_after=0.6,
                                        **job_kwargs)

                                        
Currently, following methods are implemented:

  * 'center_of_mass' 
  * 'monopolar_triangulation' with optimizer='least_square'
    This methid is from Julien Boussard, Erdem Varol and Charlie Windolf from Paninski lab.
  * 'monopolar_triangulation' with optimizer='minimize_with_log_penality'

Theses methods are the same implemented in :code:`spieinterface.toolkit.postprocessing.unit_localization`



The output :code:`peak_locations` is a 1d numpy array with a dtype that depend on the choosen method.

For instance 'monopolar_triangulation' method will have:

.. code-block:: python

    localization_dtype = [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')]

.. note::

   By convention in spikeinterface, when a probe is describe in 2d
     * **'x'** is the width of the probe
     * **'y'** is the depth
     * **'z'** is the orthogonal to the probe plane


Peak selection
--------------

When too much peaks are detected a strategy can be to select only some of then before clustering.
This is the strategy used by spyking-circus or tridesclous for instance.
Then the template are extracted from theses sub selection and a template matching step can be run.

The way the *peak vector* is reduce (aka sampled) is a crutial step because units with small firing rate
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

Recently drift estimation have been added in the sorting pipeline.
Neuropixel datsets have shown that this is crucials step.

Several methods have been proposed for this. Only one is implemented in spikeinterface at the moment.
See `Decentralized Motion Inference and Registration of Neuropixel Data <https://ieeexplore.ieee.org/document/9414145>`_
This steps is after peak detection and peak localization.
It divide the duration in time bin and estimate the relative motion in between temporal bins.

This methods have 2 flavor:

  * rigid drift : on motion vector for the entire probe is estimated
  * non rigid drift : one motion vector per depth bins

Here an example with non rigid motion estimation
  
.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    peaks = detect_peaks(recording, ...)
    
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    peak_locations = localize_peaks(recording, peaks, ...)
    
    motion, temporal_bins, spatial_bins,
                extra_check = estimate_motion(recording, peaks, peak_locations=peak_locations,
                                              direction='y', bin_duration_s=1., bin_um=10., 
                                              margin_um=5,
                                              method='decentralized_registration', 
                                              method_kwargs={},
                                              non_rigid_kwargs={
                                                  'bin_step_um': 50},
                                              output_extra_check=True,
                                              progress_bar=True, 
                                              verbose=True)    
In this example, because it is a non rigid estimation, :code:`motion` is a 2d array (num_time_bin, num_spatial_bin)


Motion correction
-----------------




Clustering
----------


Template matching
-----------------


