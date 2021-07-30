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

For now, we have methods for peak detection and peak localization. We are going to port methods for drift-correction,
clustering, template-matching, and postprocessing/cleaning in the future.


Peak detection
--------------

Peak detection is usually the first step of spike sorting and it consists of finding peaks in the traces that could
be actual spikes.

Peaks can be detected with the :code:`detect_peaks()` function as follows:

.. code-block:: python

    import spikeinterface.sortingcomponents as scp

    peaks = scp.detect_peaks(recording, method='by_channel',
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

Currently, only the "center of  mass" method is implemented.

Peak localization can be run as follows:

.. code-block:: python

    import spikeinterface.sortingcomponents as scp

    peak_locations = scp.localize_peaks(recording, peaks, method='center_of_mass',
                                        local_radius_um=150, ms_before=0.3, ms_after=0.6,
                                        **job_kwargs)


The output :code:`peak_locations` is a numpy array with dimension (num_spikes, 2), where the second dimension represent
the x-y axis.


Drift correction
----------------

**COMING SOON**

Clustering
----------

**COMING SOON**

Template matching
-----------------

**COMING SOON**

Postprocessing
--------------

**COMING SOON**
