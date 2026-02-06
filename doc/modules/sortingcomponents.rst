.. _sorting-components-module:

Sorting Components module
=========================

Spike sorting is comprised of several steps, or components. In the :py:mod:`spikeinterface.sortingcomponents` module we
are building a library of methods and steps that can be assembled to build full spike sorting pipelines.

The goal is to allow for the *modularization* of spike sorting algorithms. Currently, spike sorters are shipped
as full packages with all the steps needed to perform end-to-end spike sorting.

However, this might not be the best option. It is in fact very likely that a sorter has one excellent step,
say the clustering, but another step, which is sub-optimal. Decoupling different steps as separate components would allow
one to mix-and-match sorting steps from different sorters.

Another advantage of *modularization* is that we can accurately benchmark every step of a spike sorting pipeline.
For example, what is the performance of peak detection method 1 or 2, provided that the rest of the pipeline is the
same?

Currently, we have methods for:

 * **peak detection**
 * **peak localization**
 * **peak selection**
 * **motion estimation**
 * **motion interpolation**
 * **clustering**
 * **template matching**



An important concept is the **node pipeline** machinery, which uses the
:py:func:`~spikeinterface.core.run_node_pipeline()` function, and will be covered in the :ref:`node-pipelines` section.

You can also have a look `spikeinterface <https://github.com/samuelgarcia/sorting_components_benchmark_paper>`_
where there are more detailed notebooks on sorting components.


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
        recording=recording,
        method='by_channel',
        peak_sign='neg',
        detect_threshold=5,
        exclude_sweep_ms=0.2,
        noise_levels=None,
        random_chunk_kwargs={},
        job_kwargs=job_kwargs,
    )

The output :code:`peaks` is a NumPy array with a length of the number of peaks found and the following dtype:

.. code-block:: python

    peak_dtype = [('sample_index', 'int64'), ('channel_index', 'int64'), ('amplitude', 'float64'), ('segment_index', 'int64')]


There are two different methods available with the :code:`method` argument:

* **'locally_exclusive'** (requires :code:`numba`): peaks on neighboring channels within a certain radius are excluded (not counted multiple times)
* **'matched_filtering'** (requires :code:`numba`): a method based on convolution by a kernel that "looks like a spike"
  at several spatial scales. This is a bit slower but can detect spikes with lower amplitude.

Other variants are also implemented (but less tested or not so useful):

* **'by_channel'** : peaks are detected separately for each channel, this should be used in high density probe layout.
* **'by_channel_torch'** (requires :code:`torch`): pytorch implementation (GPU-compatible) that uses max pooling for time deduplication
* **'locally_exclusive_torch'** (requires :code:`torch`): pytorch implementation (GPU-compatible) that uses max pooling for space-time deduplication

.. note::

    The torch implementations give slightly different results due to a different implementation.

Peak detection, as many of the other sorting components, can be run in parallel.


Peak localization
-----------------

Peak localization estimates the spike *location* on the probe. An estimate of location can be important to correct for
drift or cluster spikes into different units.


Peak localization can be run using :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks()` as
follows:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)

    peak_locations = localize_peaks(
        recording=recording,
        peaks=peaks,
        method='center_of_mass',
        method_kwargs=dict(
          radius_um=70.,
          ms_before=0.3,
          ms_after=0.6,
        ),
        job_kwargs=job_kwargs,
    )


Currently, the following methods are implemented:

  * **'center_of_mass'** : the fastest and most intuitive. This method is not accurate on the
    border of the probe, so for neuropixel only the 'y' axis will be well estimated.
    For in vitro, with a square MEA, all spikes on borders will also be biased.
  * **'monopolar_triangulation'** with optimizer='least_square'
    This method is from Julien Boussard and Erdem Varol from the Paninski lab.
    This has been presented at `NeurIPS <https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26709>`_
    see also `here <https://openreview.net/forum?id=ohfi44BZPC4>`_
    **'monopolar_triangulation'** has some variant with differents optimizers (default is 'minimize_with_log_penality')
  * **'grid_convolution'** : inspired by the Kilosort approach. This consists of a convolution of traces with waveform
     prototypes with varying local spatial footprint on the probe.


Please have a look at [Scopin2024]_, for details on these methods.


These methods are the same as implemented in :py:mod:`spikeinterface.postprocessing.unit_localization`



The output :code:`peak_locations` is a 1d NumPy array with a dtype that depends on the chosen method.

For instance, the 'monopolar_triangulation' method will have:

.. code-block:: python

    localization_dtype = [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')]

.. note::

   By convention in SpikeInterface, when a probe is described in 3d
     * **'x'** is the width of the probe
     * **'y'** is the depth
     * **'z'** is orthogonal to the probe plane


.. _node-pipelines:

Node pipelines
--------------

Both :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks()` and
:py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks()` need to walk throughout the entire
recording traces, duplicating the reading of the traces from disk and applying the preprocessing.
This can be very slow!

Luckily, there is an internal machinery to avoid the multiple passes over the traces: the :py:func:`~spikeinterface.core.run_node_pipeline()` function.

The *node pipeline* is an API that runs user-selected *nodes* in parallel on all traces' chunks and performs computations like
**peak detection**, **peak localization**, **svd featuring**, ...

Here is a small example that does peak detection and localization at once.
In the following, please note that there is an intermediate node, the `ExtractDenseWaveforms` node, that does not output final results (notice the `return_output=False`), but is needed to extract waveforms for the localization node.


.. code-block:: python

  import spikeinterface.full as si

  # generate
  recording, _, _ = si.generate_drifting_recording(
      probe_name="Neuropixels1-128",
      num_units=200,
      duration=300.,
      seed=2205,
      extra_outputs=False,
  )

  # let's makes a 3 nodes

  # Node 0 : detect peak
  noise_levels = si.get_noise_levels(recording, return_in_uV=False)
  from spikeinterface.sortingcomponents.peak_detection.method_list import LocallyExclusivePeakDetector
  node0 = LocallyExclusivePeakDetector(
      recording,
      return_output=True, # We want output from this node!!
      # then specific params
      noise_levels=noise_levels,
      peak_sign="neg",
      detect_threshold=5.,
      exclude_sweep_ms=0.5
  )

  # Node 1 : extract local waveforms
  from spikeinterface.core.node_pipeline import ExtractDenseWaveforms
  node1 = ExtractDenseWaveforms(
      recording,
      parents=[node0],
      return_output=False, # We do NOT want to output all dense waveforms!!!!
      # then specific params
      ms_before=1.,
      ms_after=1.5,
  )

  # Node 2 : localize peaks using local waveforms
  from spikeinterface.sortingcomponents.peak_localization.method_list import LocalizeMonopolarTriangulation
  node2 = LocalizeMonopolarTriangulation(
      recording,
      parents=[node0, node1],
      return_output=True, # We want output from this node!!
      # then specific params
      radius_um=75.0,
      optimizer="minimize_with_log_penality",
  )

  nodes = [node0, node1, node2]

  # our dear jobs kwargs dict
  job_kwargs = dict(n_jobs=-1, chunk_duration="500ms", progress_bar=True)

  # only 2 nodes give outputs
  from spikeinterface.core.node_pipeline import run_node_pipeline
  peaks, peak_locations = run_node_pipeline(recording, nodes, job_kwargs, job_name="my pipeline", gather_mode="memory")

  # We strongly hope that geeks from various lab will appreciate the design.
  # We spent hours debating on how to do it.


Peak selection
--------------

When too many peaks are detected a strategy can be used to select (or sub-sample) only some of them before clustering.
This is the strategy used by spyking-circus and tridesclous, for instance.
Then, clustering is run on this subset of peaks, templates are extracted, and a template-matching step is run to find
all spikes.

The way the *peak vector* is reduced (or sub-sampled) is a crucial step because units with small firing rates
can be *hidden* by this process.


.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    many_peaks = detect_peaks(...) # as in above example

    from spikeinterface.sortingcomponents.peak_selection import select_peaks

    some_peaks = select_peaks(peaks=many_peaks, method='uniform', n_peaks=10000)

Implemented methods are the following:

  * **'uniform'**
  * **'uniform_locations'**
  * **'smart_sampling_amplitudes'**
  * **'smart_sampling_locations'**
  * **'smart_sampling_locations_and_time'**



Motion estimation
-----------------

Drift estimation is implemented directly in spikeintertface. So even sorters that do not
handle drift can benefit from drift estimation/correction.
Especially for acute Neuropixels-like probes, this is a crucial step.

The motion estimation step comes after peak detection and peak localization. Read more about
it in the :ref:`motion_correction` modules doc, and a more practical guide in the
:ref:`handle-drift-in-your-recording` How To.

Here is an example with non-rigid motion estimation:

.. code-block:: python

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    peaks = detect_peaks(recording=recording, ...) # as in above example

    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    peak_locations = localize_peaks(recording=recording, peaks=peaks, ...) # as above


    from spikeinterface.sortingcomponents.motion import estimate_motion
    motion = estimate_motion(
        recording=recording,
        peaks=peaks,
        peak_locations=peak_locations,
        method="dredge_ap",
        rigid=False,
        win_shape="gaussian",
        win_step_um=200.0,
        win_scale_um=300.0,
        win_margin_um=None,
        bin_um=1.0,
        bin_s=1.0,
        direction='y',
        progress_bar=True,
        verbose=True
    )

In this example, because it is a non-rigid estimation, :code:`motion` handles a 2d array (num_time_bins, num_spatial_bins).
We could now check the ``motion`` object and see if we need to apply a correction.

Availables methods are:

  * **'dredge_ap'** : the most mature method at the moement, done by [Windolf_b]_
  * **'decentralized'** : more or less the ancestor of 'dredge_ap'
  * **'iterative_template'** : this mimics the kilosort approach.
  * **'medicine'** : a more recent approach done in [Watters]_.

A comparison of these methods can be read in [Garcia2024]_.


Motion interpolation
--------------------

The estimated motion can be used to interpolate traces to attempt to correct for drift.
One possible way is to make an interpolation sample-by-sample to compensate for the motion.
The :py:class:`~spikeinterface.sortingcomponents.motion.InterpolateMotionRecording` is a preprocessing
step doing this. This preprocessing is *lazy*, so that interpolation is done on-the-fly. However, the class needs the
"motion vector" as input, which requires a relatively long computation (peak detection, localization and motion
estimation).

Here is a short example that depends on the output of "Motion interpolation":

.. code-block:: python

  from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

  recording_corrected = InterpolateMotionRecording(
      recording=recording_with_drift,
      motion=motion,
      spatial_interpolation_method='kriging',
      border_mode='remove_channels'
  )

**Notes**:
  * :code:`spatial_interpolation_method` "kriging" or "iwd" do not play a big role.
  * :code:`border_mode` is a very important parameter. It controls dealing with the border because motion causes units on the
    border to not be present throughout the entire recording. We highly recommend the :code:`border_mode='remove_channels'`
    because this removes channels on the border that will be impacted by drift. Of course the larger the motion is
    the greater the number of channels that would be removed.


Clustering
----------

The clustering step remains the central step of spike sorting.
Historically this step was separated into two distinct parts: feature reduction and clustering.
In SpikeInterface, we decided to regroup these two steps into the same module.
This allows one to compute feature reduction 'on-the-fly' and avoid long computations and storage of
large features.

The clustering step takes the recording and detected (and optionally selected) peaks as input and returns
a label for every peak.

Some methods have been implemented with various ideas in mind. We really hope that this list will be extended
soon by talented people willing to improve it. This is a crucial and not totally resolved step.

  * **'iterative-hdbscan'** : method used in spkyking-circus2. This performs local hdbscan clusetrings on
     svd waveforms features.
  * **'iterative-isosplit'** :  method used in tridesclous2. This performs local isosplit clusetrings on
     svd waveforms features.
  * **'hdbscan-positions'** : This performs a hdbscan clusetring based on the localizations of the spikes.
    This mimics the herdingspikes approach : make the clustering on spike position only but more flexible
    because more localization methods are availables.
  * **'random-projections'** : attempt to make the feature from waveforms with random projections instead of the
    good-old-school-pca.
  * **'graph-clustering'** : attempt to resolve the clusetring globally and not locally. This constructs a global
    but sparse distance matrix between all spikes. Can be slow. Then it performs 'classical' algos on
    graph (Louvain, Leiden or even HDBSCAN). Promising method but not as efficient as the 'iterative-isosplit' or
    'iterative-hdbscan'.



.. code-block:: python

  from spikeinterface.sortingcomponents.peak_detection import detect_peaks
  peaks = detect_peaks(recording, ...) # as in above example

  from spikeinterface.sortingcomponents.clustering import find_clusters_from_peaks
  labels, peak_labels = find_clusters_from_peaks(recording=recording, peaks=peaks, method="iterative-isosplit")


* **labels** : contains all possible labels (aka unit_ids)
* **peak_labels** : vector with the same size as peaks containing the label for each peak


Extract SVD from peaks
----------------------


Importantly many clustering functions internally use the
:py:func:`~spikeinterface.sortingcomponents.clusetring.extract_peaks_svd.extract_peaks_svd()`.
This runs a **node pipeline** on a selected peaks set that extracts waveforms, sparsifies them, and compresses
them on the time axis using **svd**.


Template matching
-----------------

Template matching is the final step used in many sorters (Kilosort, SpyKING-Circus, YASS, Tridesclous, HDsort...)

In this step, from a given catalogue (or dictionary) of templates (or atoms), the algorithms try to *explain* the
traces as a linear sum of a template plus a residual noise.

At the moment, there are five methods implemented:

  * **'nearest'**: a simple implementation which is more or less a np.argmin distance for the spike waveforms against all templates.
  * **'nearest-svd'**: a smarter implementation than 'nearest' using svd compression and spatial sparsity.
  * **'tdc-peeler'**: a simple idea similar to 'nearest'. Perform nearest on locally detected peaks, fit the amplitudes and
    remove them from the traces. Then re-run on residual. A bit naive but this is very fast.
  * **'circus-omp'**: a more serious implementation orthogonal template matching. This internally make a convolution
    of traces with all templates with some svd decomposition tricks to be faster. This is quite accurate but
    need lots of memory.
  * **'wobble'**: this is a re-implementation of the yass template matching code. Also very similar to 'circus-omp'.
    This is the most accurate methods for discovering spike collisions.
