Exporter module
===============

The :py:mod:`spikeinterface.exporters` module includes functions to export SI objects to other commonly used frameworks.


Exporting to Phy
----------------

The :py:func:`~spikeinterface.exporters.export_to_phy()` function allows you to use the
`Phy template GUI <https://github.com/cortex-lab/phy>`_ for visual inspection and manual curation of spike sorting
results.

**Note** : :py:func:`~spikeinterface.exporters.export_to_phy()` speed and the size of the folder will highly depends
on the sparsity of the :code:`WaveformExtractor` itself or the external specified sparsity.
The phy viewer enable to explore pca, spike amplitudes and quality metrics. So if this extension as already computed
(see :ref:`waveform_extensions``) then export should fast and also the user have better control on the parameters.
If not computed, then theses :code:`spike_amplitudes`, :code:`principal_components` can be forced


The input of the :py:func:`~spikeinterface.exporters.export_to_phy()` is a :code:`WaveformExtractor` object.

.. code-block:: python

    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_principal_components
    from spikeinterface.exporters import export_to_phy

    # the waveforms is sparse so fasten the export to phy
    folder = 'waveforms_mearec'
    we = extract_waveforms(recording, sorting, folder, sparse=True)

    # some computation are done before to control options
    compute_spike_amplitudes(we)
    compute_principal_components(we, n_components=3, mode='by_channel_global')

    # the export process is fast because everything is pre computed in advance
    export_to_phy(we, output_folder='/path/to/phy_folder')



Export a spike sorting report
-----------------------------

TODO explain sparsity
TODO explain WE extension


The :py:func:`~spikeinterface.exporters.export_report()`  provides an overview of the spike sorting output.
The report is a simple folder that contains figures (in png, svg or pdf format) and tables (csv) that can be easily
explore without any gui.
It is designed to be a common and shareable report to assess spike sorting quality with students,
collaborators, and journals.
$
The report includes summary figures of the  spike sorting output (e.g. amplitude distributions, unit localization and
depth VS amplitude) as well as unit-specific reports, that include waveforms, templates, template maps,
ISI distributions, and more.

**Note** : similarly to :py:func:`~spikeinterface.exporters.export_to_phy()` the  :py:func:`~spikeinterface.exporters.export_report()`
depends on the sparsity of the :code:`WaveformExtractor` itself and and wich extensions have been computed.
And so :code:`spike_amplitudes` and :code:`correlograms` will be automatically included in the report 
if they are computed in advance. The function can force the computation but it is a better practice to compute everything before.

Note that every units will generate a figure so the export process can be slow!

.. code-block:: python

    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms
    from spikeinterface.qualitymetrics import compute_quality_metrics
    from spikeinterface.exporters import export_report


    # the waveforms is sparse for better figures
    we = extract_waveforms(recording, sorting, folder='/path/to/wf', sparse=True)

    # some computation are done before to control options
    compute_spike_amplitudes(we)
    compute_correlograms(we)
    compute_quality_metrics(we, metric_names=['snr', 'isi_violation', 'presence_ratio'])

    # the export process 
    export_report(we, output_folder='/path/to/phy_folder')
