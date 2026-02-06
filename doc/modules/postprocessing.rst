Postprocessing module
=====================

SpikeInterface's postprocessing is centered around the :py:class:`~spikeinterface.core.SortingAnalyzer`
object. This combines recording and sorting information, and can be used to compute lots of information
you might be interested in: waveforms, templates, correlograms, spike amplitudes and more. We hope this
object can also serve as a basis for other tools, and is already used by SpikeInterface-GUI, SortingView
and NeuroConv.

Learn how to make a SortingAnalyzer in `our tutorial
<https://spikeinterface.readthedocs.io/en/latest/tutorials/core/plot_4_sorting_analyzer.html>`_

.. _extensions:

After spike sorting, we can use the :py:mod:`~spikeinterface.postprocessing` module to further post-process
the spike sorting output. First, make a :py:class:`~spikeinterface.core.SortingAnalyzer` object. We can
then compute "extensions" of the sorting analyzer. If you'd like to play with the extensions on a
simulated analyzer, we can create a simulated one as follows:

.. code-block:: python

    import spikeinterface.core as si

    recording, sorting = si.generate_ground_truth_recording()
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)


.. _waveform_extensions:

Extensions as AnalyzerExtensions
--------------------------------

There are several postprocessing tools available, and all of them are implemented as a
:py:class:`~spikeinterface.core.AnalyzerExtension`. If the :code:`SortingAnalyzer` is saved to disk, all computations on
top of it will be saved alongside the :code:`SortingAnalyzer` itself (sub folder, zarr path or sub dict).
This workflow is convenient for retrieval of time-consuming computations (such as pca or spike amplitudes) when reloading a
:code:`SortingAnalyzer`.

:py:class:`~spikeinterface.core.AnalyzerExtension` objects are tightly connected to the
parent :code:`SortingAnalyzer` object, so that operations done on the :code:`SortingAnalyzer`, such as saving,
loading, or selecting units, will be automatically applied to all extensions.

To check what extensions have already been calculated for a :code:`SortingAnalyzer` named :code:`sorting_analyzer`, you can use:

.. code-block:: python

    available_extension_names = sorting_analyzer.get_loaded_extension_names()
    print(available_extension_names)

.. code-block:: bash

    >>> ["principal_components", "spike_amplitudes"]

In this case, for example, principal components and spike amplitudes have already been computed.
To load the extension object you can run:

.. code-block:: python

    ext = sorting_analyzer.get_extension("spike_amplitudes")
    ext_data = ext.get_data()

Here :code:`ext` is the extension object (in this case the :code:`ComputeSpikeAmplitudes`), and :code:`ext_data` will
contain the actual amplitude data. Note that different extensions might have different ways to return the extension.
You can use :code:`ext.get_data?` for documentation.


To check what extensions spikeinterface can calculate, you can use the :code:`get_computable_extensions` method:

.. code-block:: python

    all_computable_extensions = sorting_analyzer.get_computable_extensions()
    print(all_computable_extensions)

.. code-block:: bash

    >>> ['random_spikes', 'waveforms', 'templates', 'noise_levels', 'amplitude_scalings', 'correlograms', 'isi_histograms', 'principal_components', 'spike_amplitudes', 'spike_locations', 'template_metrics', 'template_similarity', 'unit_locations', 'quality_metrics']

There is detailed documentation about each extension :ref:`below<modules/postprocessing:Available postprocessing extensions>`.
Each extension comes from a different module. To use the :code:`postprocessing` extensions, you'll need to have the `postprocessing`
module loaded.

Some extensions depend on another extension. For instance, you can only calculate `principal_components` if you've already calculated
both `random_spikes` and `waveforms`. We say that `principal_components` is a child of the other two or that is *depends* on the other
two. Other extensions, like `isi_histograms`, don't depend on anything. It has no children and no parents. The parent/child
relationships of all the extensions currently defined in spikeinterface can be found in this diagram:

.. figure:: ../images/parent_child.svg
    :alt: Parent child relationships for the extensions in spikeinterface
    :align: center

If you try to calculate a child before calculating a parent, an error will be thrown. Further, when a parent is recalculated we delete
its children. Why? Consider calculating :code:`principal_components`. This depends on random selection of spikes chosen
during the computation of :code:`random_spikes`. If you recalculate the random spikes, a different selection will be chosen and your
:code:`principal_components` will change (a little bit). Hence your principal components are inconsistent with the random spikes. To
avoid this inconsistency, we delete the children.

We can also delete an extension ourselves:

.. code-block:: python

    sorting_analyzer.delete_extension("spike_amplitudes")

This does *not* delete the children of the extension, since there are some cases where you might want to delete e.g. the (large)
waveforms but keep the (smaller) postprocessing outputs.

Computing extensions
--------------------

To compute extensions we can use the :code:`compute` method. There are several ways to pass parameters so we'll go through them here,
focusing on the :code:`principal_components` extension. Here's one way to compute
the principal components of a :code:`SortingAnalyzer` object called :code:`sorting_analyzer` with default parameters:

.. code-block:: python

    sorting_analyzer.compute("principal_components")

In this simple case you can alternatively use :code:`compute_principal_components(sorting_analyzer)`, which matches legacy syntax.
You can also compute several extensions at the same time by passing a list:

.. code-block:: python

    sorting_analyzer.compute(["random_spikes", "waveforms", "principal_components"])

You might want to change the parameters. Two parameters of principal_components are :code:`n_components` and :code:`mode`.
We can choose these as follows:

.. code-block:: python

    sorting_analyzer.compute("principal_components", n_components=3, mode="by_channel_local")

As your code gets more complicated it might be easier to store your calculation in a dictionary, especially if you're calculating more
than one thing:

.. code-block:: python

    compute_dict = {
        'random_spikes': {},
        'principal_components': {'n_components': 3, 'mode': 'by_channel_local'},
        'templates': {'operators': ["average"]},
    }
    sorting_analyzer.compute(compute_dict)

There are also hybrid options, which can be helpful if you're mostly using default parameters:

.. code-block:: python

    # here `templates` will be calculated using default parameters.
    extension_params = {
        'principal_components': {'n_components': 3, 'mode': 'by_channel_local'},
    }
    sorting_analyzer.compute(
        ["principal_components", "templates"],
        extension_params=extension_params
    )

Extensions are generally saved in two ways, suitable for two workflows:

1. When the sorting analyzer is stored in memory, the extensions are only saved when the :code:`.save_as` method is called.
   This saves the sorting analyzer and all it's extensions in their current state. This is useful when trying out different
   parameters and initially setting up your pipeline.
2. When the sorting analyzer is stored on disk the extensions are, by default, saved when they are calculated. You calculate
   extensions without saving them by specifying :code:`save=False` as a :code:`compute` argument. (e.g.
   :code:`sorting_analyzer.compute('waveforms', save=False)`).


.. note::

    We recommend choosing a workflow and sticking with it. Either keep everything on disk or keep everything in memory until
    you'd like to save. A mixture can lead to unexpected behavior. For example, consider the following code

.. code::

    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
    )

    sorting_analyzer.save_as(folder="my_sorting_analyzer")
    sorting_analyzer.compute("random_spikes", save=True)

Here the random_spikes extension is **not** saved. The :code:`sorting_analyzer` is **still** saved in memory. The :code:`save_as` method only made a snapshot
of the sorting analyzer which is saved in a folder. Hence :code:`compute` doesn't know about the folder
and doesn't save anything. If we wanted to save the extensions as we compute them, we should have started with a "binary_folder" sorting analyzer:

.. code::

    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder="my_sorting_analyzer"
    )

    sorting_analyzer.compute("random_spikes", save=True)


Available postprocessing extensions
-----------------------------------

.. _postprocessing_noise_levels:

noise_levels
^^^^^^^^^^^^

This extension computes the noise level of each channel using the median absolute deviation.
As an extension, this expects the :code:`Recording` as input and the computed values are persistent on disk.

.. code-block:: python

    # compute the noise from an analyzer
    noise = sorting_analyzer.compute("noise_levels")


.. _postprocessing_principal_components:

principal_components
^^^^^^^^^^^^^^^^^^^^

This extension computes the principal components of the waveforms. There are several modes available:

* "by_channel_local" (default): fits one PCA model for each by_channel
* "by_channel_global": fits the same PCA model to all channels (also termed temporal PCA)
* "concatenated": concatenates all channels and fits a PCA model on the concatenated data

If the input :code:`SortingAnalyzer` is sparse, the sparsity is used when computing the PCA.
For dense waveforms, sparsity can also be passed as an argument.

.. code-block:: python

    pc = sorting_analyzer.compute(
        input="principal_components",
        n_components=3,
        mode="by_channel_local"
    )

For more information, see :py:func:`~spikeinterface.postprocessing.compute_principal_components`

template_similarity
^^^^^^^^^^^^^^^^^^^

This extension computes the similarity of the templates to each other. This information could be used for automatic
merging. Currently, the only available similarity method is the cosine similarity, which is the angle between the
high-dimensional flattened template arrays. Note that cosine similarity does not take into account amplitude differences
and is not well suited for high-density probes.

.. code-block:: python

    similarity = sorting_analyzer.compute(input="template_similarity", method='cosine_similarity')


For more information, see :py:func:`~spikeinterface.postprocessing.compute_template_similarity`

.. _postprocessing_spike_amplitudes:

spike_amplitudes
^^^^^^^^^^^^^^^^

This extension computes the amplitude of each spike as the value of the traces on the extremum channel at the times of
each spike. The extremum channel is computed from the templates.


**NOTE:** computing spike amplitudes is highly recommended before calculating amplitude-based quality metrics, such as
:ref:`amp_cutoff` and :ref:`amp_median`.

.. code-block:: python

    amplitudes = sorting_analyzer.compute(input="spike_amplitudes")

For more information, see :py:func:`~spikeinterface.postprocessing.compute_spike_amplitudes`


.. _postprocessing_amplitude_scalings:

amplitude_scalings
^^^^^^^^^^^^^^^^^^

This extension computes the amplitude scaling of each spike as the value of the linear fit between the template and the
spike waveform. In case of spatio-temporal collisions, a multi-linear fit is performed using the templates of all units
involved in the collision.

**NOTE:** computing amplitude scalings is highly recommended before calculating amplitude-based quality metrics, such as
:ref:`amp_cutoff` and :ref:`amp_median`.

.. code-block:: python

    amplitude_scalings = sorting_analyzer.compute(input="amplitude_scalings")

For more information, see :py:func:`~spikeinterface.postprocessing.compute_amplitude_scalings`

.. _postprocessing_spike_locations:

spike_locations
^^^^^^^^^^^^^^^


This extension estimates the location of each spike in the sorting output. Spike location estimates can be done
with center of mass (:code:`method="center_of_mass"` - fast, but less accurate), a monopolar triangulation
(:code:`method="monopolar_triangulation"` - slow, but more accurate), or with the method of grid convolution
(:code:`method="grid_convolution"`)

**NOTE:** computing spike locations is required to compute :ref:`drift_metrics`.

.. code-block:: python

    spike_locations = sorting_analyzer.compute(
        input="spike_locations",
        ms_before=0.5,
        ms_after=0.5,
        spike_retriver_kwargs=dict(
            channel_from_template=True,
            radius_um=50,
            peak_sign="neg"
        ),
        method="center_of_mass"
    )


For more information, see :py:func:`~spikeinterface.postprocessing.compute_spike_locations`


unit_locations
^^^^^^^^^^^^^^


This extension is similar to the :code:`spike_locations`, but instead of estimating a location for each spike
based on individual waveforms, it calculates at the unit level using templates. The same localization methods
(:code:`method="center_of_mass" | "monopolar_triangulation" | "grid_convolution"`) are available.


.. code-block:: python

    unit_locations = sorting_analyzer.compute(input="unit_locations", method="monopolar_triangulation")

For more information, see :py:func:`~spikeinterface.postprocessing.compute_unit_locations`


correlograms
^^^^^^^^^^^^

This extension computes correlograms (both auto- and cross-) for spike trains. The computed output is a 3d array
with shape (num_units, num_units, num_bins) with all correlograms for each pair of units (diagonals are auto-correlograms).

.. code-block:: python

    ccg = sorting_analyzer.compute(
        input="correlograms",
        window_ms=50.0,
        bin_ms=1.0,
        method="auto"
    )

For more information, see :py:func:`~spikeinterface.postprocessing.compute_correlograms`


acgs_3d
^^^^^^^

This extension computes the 3D Autocorrelograms (3D-ACG) from units' spike times to analyze how a neuron's temporal
firing pattern varies with its firing rate. The 3D-ACG, described in [Beau]_ et al., 2025, provides rich
representations of a unit's spike train statistics while accounting for firing rate modulations.

.. code-block:: python

    acg3d = sorting_analyzer.compute(
        input="acgs_3d",
        window_ms=50.0,
        bin_ms=1.0,
        num_firing_rate_quantiles=10,
        smoothing_factor=250,
    )

For more information, see :py:func:`~spikeinterface.postprocessing.compute_acgs_3d`


isi_histograms
^^^^^^^^^^^^^^

This extension computes the histograms of inter-spike-intervals. The computed output is a 2d array with shape
(num_units, num_bins), with the isi histogram of each unit.


.. code-block:: python

    isi =  sorting_analyzer.compute(
        input="isi_histograms",
        window_ms=50.0,
        bin_ms=1.0,
        method="auto"
    )

valid_unit_periods
^^^^^^^^^^^^^^^^^^

This extension computes the valid unit periods for each unit based on the estimation of false positive rates
(using RP violation - see ::doc:`metrics/qualitymetrics/isi_violations`) and false negative rates
(using amplitude cutoff - see ::doc:`metrics/qualitymetrics/amplitude_cutoff`) computed over chunks of the recording.
The valid unit periods are the periods where both false positive and false negative rates are below specified
thresholds. Periods can be either absolute (in seconds), same for all units, or relative, where
chunks will be unit-specific depending on firing rate (with a target number of spikes per chunk).

.. code-block:: python

    valid_periods = sorting_analyzer.compute(
        input="valid_unit_periods",
        period_mode='relative',
        target_num_spikes=300,
        fp_threshold=0.1,
        fn_threshold=0.1,
    )

For more information, see :py:func:`~spikeinterface.postprocessing.compute_valid_unit_periods`.




Other postprocessing tools
--------------------------

align_sorting
^^^^^^^^^^^^^

This function aligns the spike trains :code:`BaseSorting` object using pre-computed shifts of misaligned templates.
To compute shifts, one can use the :py:func:`~spikeinterface.core.get_template_extremum_channel_peak_shift` function.

For more information, see :py:func:`~spikeinterface.postprocessing.align_sorting`
