Exporters module
================

The :py:mod:`spikeinterface.exporters` module includes functions to export SpikeInterface objects to other commonly
used frameworks.

Exporting to Pynapple
---------------------

The Python package `Pynapple <https://pynapple.org/>`_ is often used for combining ephys
and behavioral data. It can be used to decode behavior, make tuning curves, compute spectrograms, and more!
The :py:func:`~spikeinterface.exporters.to_pynapple_tsgroup` function allows you to convert a
SortingAnalyzer to Pynapple's ``TsGroup`` object on the fly.

.. note::

    When creating the ``TsGroup``, we will use the underlying time support of the SortingAnalyzer.
    How this works depends on your acquisition system. You can use the ``get_times`` method on a recording
    (``my_recording.get_times()``) to find the time support of your recording.

When constructed, if ``attach_unit_metadata`` is set to ``True``, any relevant unit information
is propagated to the ``TsGroup``. The ``to_pynapple_tsgroup`` checks if unit locations, quality
metrics and template metrics have been computed. Whatever has been computed is attached to the
returned object. For more control, set ``attach_unit_metadata`` to ``False`` and attach metadata
using ``Pynapple``'s ``set_info`` method.

The following code creates a ``TsGroup`` from a ``SortingAnalyzer``, then saves it using ``Pynapple``'s
save method.

.. code-block:: python

    import spikeinterface as si
    from spikeinterface.exporters import to_pynapple_tsgroup

    # load in an analyzer
    analyzer = si.load_sorting_analyzer("path/to/analyzer")

    my_tsgroup = to_pynapple_tsgroup(
        sorting_analyzer=analyzer,
        attach_unit_metadata=True,
    )

    # Note: can add metadata using e.g.
    # my_tsgroup.set_info({'brain_region': ['MEC', 'MEC', ...]})

    my_tsgroup.save("my_tsgroup_output.npz")

If you have a multi-segment sorting, you need to pass the ``segment_index`` argument to the
``to_pynapple_tsgroup`` function. This way, you can generate one ``TsGroup`` per segment.
You can later concatenate these ``TsGroup`` s using Pynapple's ``concatenate`` functionality.

Exporting to Phy
----------------

The :py:func:`~spikeinterface.exporters.export_to_phy` function allows you to use the
`Phy template GUI <https://github.com/cortex-lab/phy>`_ for visual inspection and manual curation of spike sorting
results.

.. note::

    :py:func:`~spikeinterface.exporters.export_to_phy` speed and the size of the folder will highly depend
    on the sparsity of the :code:`SortingAnalyzer` itself or the external specified sparsity.
    The Phy viewer enables one to explore PCA projections, spike amplitudes, waveforms and quality of spike sorting results.
    So if these pieces of information have already been computed as extensions (see :ref:`modules/postprocessing:Extensions as AnalyzerExtensions`),
    then exporting to Phy should be fast (and the user has better control of the parameters for the extensions).
    If not pre-computed, then the required extensions (e.g., :code:`spike_amplitudes`, :code:`principal_components`)
    can be computed directly at export time.

The input of the :py:func:`~spikeinterface.exporters.export_to_phy` is a :code:`SortingAnalyzer` object.

.. code-block:: python

    import spikeinterface as si # core module only
    from spikeinterface.exporters import export_to_phy

    # the waveforms are sparse so it is faster to export to phy
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)

    # some computations are done before to control all options
    sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    _ = sorting_analyzer.compute('spike_amplitudes')
    _ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")

    # the export process is fast because everything is pre-computed
    export_to_phy(sorting_analyzer=sorting_analyzer, output_folder='path/to/phy_folder')


Export to IBL GUI
-----------------

The :py:func:`~spikeinterface.exporters.export_to_ibl_gui` function allows you to use the
`IBL GUI <https://github.com/int-brain-lab/iblapps/wiki>`_ for probe alignment.

The IBL GUI can also be installed as a standalone app using `this fork <https://github.com/AllenNeuralDynamics/ibl-ephys-alignment-gui>`_ from the Allen Institute.

The input of the :py:func:`~spikeinterface.exporters.export_to_ibl_gui` is a :code:`SortingAnalyzer` object.

.. code-block:: python

    import spikeinterface as si # core module only
    import spikeinterface.preprocessing as spre
    from spikeinterface.exporters import export_to_ibl_gui

    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)

    # we need to compute some required extensions
    sorting_analyzer.compute(['random_spikes', 'templates', 'spike_amplitudes', 'spike_locations', 'noise_levels', 'quality_metrics'])
    # note that spike_locations are optional, but recommended to compute accurate spike depths

    # optionally, we can pass an LFP recording to compute RMS/PSD in the LFP band
    recording_lfp = spre.bandpass_filter(recording, freq_min=1, freq_max=300)
    # we can also decimate the LFP to speed up the process
    recording_lfp = spre.decimate(recording_lfp, 10)

    # the export process is fast because everything is pre-computed
    export_to_ibl_gui(
        sorting_analyzer=sorting_analyzer,
        output_folder='path/to/ibl_folder',
        lfp_recording=recording_lfp,
        n_jobs=-1
    )


Export a spike sorting report
-----------------------------


The :py:func:`~spikeinterface.exporters.export_report`  provides an overview of the spike sorting output.
The report is a simple folder that contains figures (in png, svg or pdf format) and tables (csv) that can be easily
explored without any GUI.
It is designed to be a common and shareable report to assess spike sorting quality with students,
collaborators, and journals.

The report includes summary figures of the spike sorting output (e.g. amplitude distributions, unit localization and
depth VS amplitude) as well as unit-specific reports, that include waveforms, templates, template maps,
ISI distributions, and more.

.. note::

    Similarly to :py:func:`~spikeinterface.exporters.export_to_phy` the
    :py:func:`~spikeinterface.exporters.export_report` depends on the sparsity of the :code:`SortingAnalyzer` itself and
    on which extensions have been computed. For example, :code:`spike_amplitudes` and :code:`correlograms` related plots
    will be automatically included in the report if the associated extensions are computed in advance.
    The function can perform these computations as well, but it is a better practice to compute everything that's needed
    beforehand.

Note that every unit will generate a summary unit figure, so the export process can be slow for spike sorting outputs
with many units!

.. code-block:: python

    import spikeinterface as si # core module only
    from spikeinterface.exporters import export_report


    # the waveforms are sparse for more interpretable figures
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording,)

    # some computations are done before to control all options
    sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    sorting_analyzer.compute(['spike_amplitudes', 'correlograms', 'template_similarity', 'quality_metrics'],
                             extension_params=dict(quality_metrics=dict(metric_names=['snr', 'isi_violation', 'presence_ratio']))
                             )

    # the export process
    export_report(sorting_analyzer=sorting_analyzer, output_folder='path/to/spikeinterface-report-folder')
