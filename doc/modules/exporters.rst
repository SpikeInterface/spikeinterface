Exporters module
================

The :py:mod:`spikeinterface.exporters` module includes functions to export SpikeInterface objects to other commonly
used frameworks.


Exporting to Phy
----------------

The :py:func:`~spikeinterface.exporters.export_to_phy` function allows you to use the
`Phy template GUI <https://github.com/cortex-lab/phy>`_ for visual inspection and manual curation of spike sorting
results.

**Note** : :py:func:`~spikeinterface.exporters.export_to_phy` speed and the size of the folder will highly depend
on the sparsity of the :code:`SortingAnalyzer` itself or the external specified sparsity.
The Phy viewer enables one to explore PCA projections, spike amplitudes, waveforms and quality of spike sorting results.
So if these pieces of information have already been computed as extensions (see :ref:`modules/postprocessing:Extensions as AnalyzerExtensions`),
then exporting to Phy should be fast (and the user has better control of the parameters for the extensions).
If not pre-computed, then the required extensions (e.g., :code:`spike_amplitudes`, :code:`principal_components`)
can be computed directly at export time.

The input of the :py:func:`~spikeinterface.exporters.export_to_phy` is a :code:`SortingAnalyzer` object.

.. code-block:: python

    import spikeinterface as si # core module only
    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_principal_components
    from spikeinterface.exporters import export_to_phy

    # the waveforms are sparse so it is faster to export to phy
    sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)

    # some computations are done before to control all options
    sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    _ = sorting_analyzer.compute('spike_amplitudes')
    _ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")

    # the export process is fast because everything is pre-computed
    export_to_phy(sorting_analyzer=sorting_analyzer, output_folder='path/to/phy_folder')



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

**Note** : similarly to :py:func:`~spikeinterface.exporters.export_to_phy` the
:py:func:`~spikeinterface.exporters.export_report` depends on the sparsity of the :code:`SortingAnalyzer` itself and
on which extensions have been computed. For example, :code:`spike_amplitudes` and :code:`correlograms` related plots
will be automatically included in the report if the associated extensions are computed in advance.
The function can perform these computations as well, but it is a better practice to compute everything that's needed
beforehand.

Note that every unit will generate a summary unit figure, so the export process can be slow for spike sorting outputs
with many units!

.. code-block:: python

    import spikeinterface as si # core module only
    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms
    from spikeinterface.qualitymetrics import compute_quality_metrics
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
