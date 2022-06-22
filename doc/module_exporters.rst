Exporter module
===============

The :py:mod:`spikeinterface.exporters` module includes functions to export SI objects to other commonly used frameworks.


Exporting to Phy
----------------

The :py:func:`~spikeinterface.exporters.export_to_phy()` function allows you to use the
`Phy template GUI <https://github.com/cortex-lab/phy>`_ for visual inspection and manual curation of spike sorting
results.

The input of the :py:func:`~spikeinterface.exporters.export_to_phy()` is a :code:`WaveformExtractor` object.

.. code-block:: python

    import spikeinterface as si
    from spikeinterface.exporters import export_to_phy

    folder = 'waveforms_mearec'
    we = si.extract_waveforms(recording, sorting, folder)

    output_folder = 'phy_folder'
    st.export_to_phy(we, output_folder)


Export a spike sorting report
-----------------------------

The **SI report** provides an overview of the spike sorting output. It is designed to be a common and shareable report
to assess spike sorting quality with students, collaborators, and journals.

The report includes summary figures of the  spike sorting output (e.g. amplitude distributions, unit localization and
depth VS amplitude) as well as unit-specific reports, that include waveforms, templates, template maps,
ISI distributions, and more.

.. code-block:: python

    import spikeinterface as si
    from spikeinterface.exporters import export_report

    folder = 'waveforms_mearec'
    we = si.extract_waveforms(recording, sorting, folder)

    output_folder = 'report_folder'
    export_report(we, output_folder)
