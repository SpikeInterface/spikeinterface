Import Kilosort4 output
=======================

If you have sorted your data with `Kilosort4 <https://github.com/MouseLand/Kilosort>`__, your sorter output is saved in format which was
designed to be compatible with `phy <https://github.com/cortex-lab/phy>`__. SpikeInterface provides a function which can be used to
transform this output into a ``SortingAnalyzer``. This is helpful if you'd like to compute some more properties of your sorting
(e.g. quality and template metrics), or if you'd like to visualize your output using `spikeinterface-gui <https://github.com/SpikeInterface/spikeinterface-gui/>`__.

To create an analyzer from a Kilosort4 output folder, simply run

.. code::

    from spikeinterface.extractors import kilosort_output_to_analyzer
    sorting_analyzer = kilosort_output_to_analyzer('path/to/output')

The ``'path/to/output'`` should point to the Kilosort4 output folder. If you ran Kilosort4 natively, this is wherever you asked Kilosort4 to
save your output. If you ran Kilosort4 using SpikeInterface, this is in the ``sorter_output`` folder inside the ``output_folder`` created
when you ran ``run_sorter``.

The ``analyzer`` object contains as much information as it can grab from the Kilosort4 output. If everything works, it should contain
information about the ``templates``, ``spike_locations`` and ``spike_amplitudes``. These are stored as ``extensions`` of the ``SortingAnalyzer``.
You can compute extra information about the sorting using the ``compute`` method. For example,

.. code::

        sorting_analyzer.compute({
            "unit_locations": {},
            "correlograms": {},
            "template_similarity": {},
            "isi_histograms": {},
            "template_metrics": {include_multi_channel_metrics: True},
            "quality_metrics": {},
        })

widgets.html#available-plotting-functions

Learn more about the ``SortingAnalyzer`` and its ``extensions`` `here <https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html>`__.

If you'd like to store the information you've computed, you can save the analyzer:

.. code::

    sorting_analyzer.save_as(
        format="binary_folder",
        folder="my_kilosort_analyzer"
    )

You now have a fully functional ``SortingAnalyzer`` - congrats! You can now use `spikeinterface-gui <https://github.com/SpikeInterface/spikeinterface-gui/>`__. to view the results
interactively, or start manually labelling your units to `create an automated curation model <https://spikeinterface.readthedocs.io/en/stable/tutorials_custom_index.html#automated-curation-tutorials>`__.
