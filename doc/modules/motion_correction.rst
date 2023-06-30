.. _motion_correction:


Motion/drift correction
=======================

Overview
--------

Mechanical drifts, often observed in recordings, are currently a major issue for spike sorting. This is especially striking
with the new generation of high-density devices used in-vivo such as the neuropixel electrodes.
The first sorter that has introduced motion/drift correction as a prepossessing step was kilosort2.5 (see [Steinmetz2021]_)

Long story short, the main idea is the same as the one used for non-rigid image registration, for example with calcium
imaging. However, because with extracellular recording we do not have a proper image to use as a reference, the main idea
of the algorithm is create an "image" via the activity profile of the cells during a given time window. Assuming this
activity profile should be kept constant over time, the motion can be estimated, by blocks, along the probe's insertion axis
(i.e. depth) so that we can interpolate the traces to compensate this estimated motion.
Users with a need to handle drift were currently forced to stick to the use of kilosort2.5 or pykilosort. Recently, the Paninski
group from Columbia university introduced a possibly more accurate method to estimate the drift (see [Varol2021]_
and [Windolf2023]_) but this new method was not properly integrated in any sorter.

Because motion registration is a hard topic, with numerous hypothesis and/or implementations details that might have a large
impact on the spike sorting performances (see [Garcia2023]_), in spikeinterface, we developed a full motion estimation
and interpolation framework to make accessible all theses methods in one place. This modular approach has offers a major benefit :
**the drift correction can be applied on a recording as a preprocessing step, and
then used for any sorter!** In short, the motion correction is decoupled from the sorter itself.

This gives the user an incredible flexibility to check/test and correct the drifts before the sorting process.

Here the overview of the motion/drift correction as a preprocessing

.. image:: ../images/motion_correction_overview.png
  :align: center

The motion correction process can be split into 3 steps:

  1. **activity profile** : detect peaks and localize them along time and depth
  2. **motion inference**: estimate the drift motion by blocks for non rigid motion
  3. **motion interpolation**: interpolate traces using the motion vector

For every steps, we implemented several methods. The combination of the yellow boxes should gives more or less what
kilosort2.5/3 is doing. Similarly, the combination of the green boxes gives the method developed by the Paninski group.
Of course the end user can combine any of the methods to get the best motion correction possible.
This make also an incredible framework for testing new ideas.

For a better overview, the spikeinterface team have publish a manuscript to validate/benchmark/compare theses motion
correction methods (see [Garcia2023]_).

Spikeinterface offers two levels for motion correction:
  1. A high level with a unique function and predefined parameters preset
  2. A low level where the user need to call one by one all functions for a better control


High level api
--------------

One challenging task for motion correction is to find parameters.
The high level :py:func:`~spikeinterface.preprocessing.correct_motion()` propose the concept of a **"preset"** that already
have predefined parameters, in order to achieve a calibrated behavior.

We propose at the moment 3 presets:

  * **"nonrigid_accurate"**: the one by Paninski group. It consists of *monopolar triangulation + decentralized + inverse distance weighted*
                             This is the slowest combination but maybe the most accurate. The main bottleneck of this preset is the monopolar
                             triangulation for the estimation of the peaks positions. To speed it up, one could think about subsampling the
                             space of all the detected peaks.
  * **"rigid_fast"**: a fast but not very accurate method. *center of mass + decentralized + inverse distance weighted*
                      To be used as check and/or control on a recording to check the presence of drift.
                      Note that, in this case the drift is considered as "rigid" over the electrode.
  * **"kilosort_like"**: this mimic what is done in kilosort. *grid convolution + iterative_template + kriging*
                         This is not exactly 100% what kilosort is doing because the peak detection is done with template
                         in kilosort and this is not case in spikeinterface. But this "preset" give similar
                         results than kilosort2.5 itself.


.. code-block:: python

  # read and preprocess
  rec = read_spikeglx('/my/Neuropixel/recording')
  rec = bandpass_filter(rec)
  rec = common_reference(rec)

  # then correction is one line of code
  rec_corrected = correct_motion(rec, preset="nonrigid_accurate")

The process is quite long due the two first steps (activity profile + motion inference)
But the return :code:`rec_corrected` is a lazy recording object this will interpolate traces on the
fly (step 3 motion interpolation).


If you want to user other preset this is just easy as

.. code-block:: python

  # mimic kilosort motion
  rec_corrected = correct_motion(rec, preset="kilosort_like")

  # super but less accurate and rigid
  rec_corrected = correct_motion(rec, preset="rigid_fast")


Optionally any parameter from the preset can be overwritten.

.. code-block:: python

    rec_corrected = correct_motion(rec, preset="nonrigid_accurate",
                                   detect_kwargs=dict(
                                       detect_threshold=10.),
                                   estimate_motion_kwargs=dic(
                                       histogram_depth_smooth_um=8.,
                                       time_horizon_s=120.,
                                   ),
                                   correct_motion_kwargs=dict(
                                        spatial_interpolation_method="kriging",
                                   )
                                   )

Importantly, all the result and intermediate computation can be saved into a folder for further loading
and checking. The folder will contain the motion vector itself of course but also detected peaks, peak location, ...


.. code-block:: python

    motion_folder = '/somewhere/to/save/the/motion'
    rec_corrected = correct_motion(rec, preset="nonrigid_accurate", folder=motion_folder)

    # and then
    motion_info = load_motion_info(motion_folder)



Low level api
-------------

All steps (**activity profile**, **motion inference**, **motion interpolation**) can be launched with distinct function.
This can be useful to find the good method and finely tune/optimize parameters at every steps.
All functions are implemented in :py:mod:`~spikeinterface.sortingcomponents`.
They all have a simple API with spikeinterface objects as inputs or numpy arrays, such that hacking should be fairly accessible.
Since motion correction is a hot topic, theses functions have many possible methods and also many possible parameters.
Finding the good combination of method/parameters is not that easy but it should be doable, assuming the presets are not
working properly for your particular case.


The high level :py:func:`~spikeinterface.preprocessing.correct_motion()` is internally equivalent to this:


.. code-block:: python

    # each import is needed
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
    from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion

    job_kwargs = dict(chunk_duration="1s", n_jobs=20, progress_bar=True)
    # Step 1 : activity profile
    peaks = detect_peaks(rec, method="locally_exclusive", detect_threshold=8.0, **job_kwargs)
    # optionally we could sub select some peak to speed up the localization
    peaks = select_peaks(peaks, ...)
    peak_locations = localize_peaks(rec, peaks, method="monopolar_triangulation",local_radius_um=75.0,
                                    max_distance_um=150.0, **job_kwargs)

    # Step 2: motion inference
    motion, temporal_bins, spatial_bins = estimate_motion(rec, peaks, peak_locations,
                                                          method="decentralized",
                                                          direction="y",
                                                          bin_duration_s=2.0,
                                                          bin_um=5.0,
                                                          win_step_um=50.0,
                                                          win_sigma_um=150.0,
                                                          )

    # Step 3: motion interpolation
    # this step is lazy
    rec_corrected = interpolate_motion(rec, motion, temporal_bins, spatial_bins,
                                       border_mode="remove_channels",
                                       spatial_interpolation_method="kriging",
                                       sigma_um=30.
    )




References
----------

.. [Steinmetz2021] `Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings <https://www.science.org/doi/10.1126/science.abf4588>`_

.. [SteinmetzDataset] `Imposed motion datasets <https://figshare.com/articles/dataset/_Imposed_motion_datasets_from_Steinmetz_et_al_Science_2021/14024495>`_

.. [Windolf2023] `Robust Online Multiband Drift Estimation in Electrophysiology Data <https://www.biorxiv.org/content/10.1101/2022.12.04.519043v2>`_

.. [Varol2021] `Decentralized Motion Inference and Registration of Neuropixel Data <https://ieeexplore.ieee.org/document/9414145>`_

.. [Garcia2023] `A modular approach to handle in-vivo drift correction for high-density extracellular recordings <https://www.biorxiv.org/content/10.1101/2023.06.29.546882v1>`_
