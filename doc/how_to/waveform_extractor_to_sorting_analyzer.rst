=========================================
From WaveformExtractor to SortingAnalyzer
=========================================

From spikeinterface version 0.101, the internal structure of the postprocessing module has
changed. We explain the motivation for the change in more detail `below <#why-change>`_.

From the user point of view, the key change is the deletion of the ``WaveformExtractor`` class and the addition of
the new ``SortingAnalyzer`` class. Hence any code using `WaveformExtractors` will have to be updated.
If you want to continue using ``WaveformExtractor`` you need to use spikeinterface versions 0.100 and
below.

Updating your old code should be straightforward. On this page, we demonstrate how to `convert old WaveformExtractor folders
to new SortingAnalyzer folders <#convert-a-waveformextractor-folder-to-a-sortinganalyzer-folder>`_ and have written a
`code dictionary <#dictionary-between-sortinganalyzer-and-waveformextractor>`_ which should make updating your codebase simple.

Why change?
^^^^^^^^^^^

Previously, a lot of post-processing was handled by the ``WaveformExtractor`` class. However, as the name suggests, this singles out waveforms as special. But many outputs don't care about waveforms. And this isn't just a naming problem: you had to create a WaveformExtractor object (and thus calculate waveforms) to calculate e.g. correlograms, which didn't depend on them. Then to access this data you had to access the WaveformExtractor. We felt that this was confusing to the user: why are the correlograms stored in a waveform class?? This mixing of waveforms with non-waveform objects also makes the codebase messy and harder to develop.

Our new class is called the ``SortingAnalyzer``. You can think of this as combining a recording and a sorting into one. There are then lots of ``extensions`` which you can calculate. Waveforms are one extension of the SortingAnalyzer, but aren't special. We hope the new class is simpler to understand, especially for beginners. It also comes with some new features: you can calculate multiple extensions at the same time, it has better caching and better zarr support (hopefully leading to fast interactive widgets in the future).

Convert a WaveformExtractor folder to a SortingAnalyzer folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several backward compatibility tools to help deal with existing ``WaveformExtractor`` folders.
For now, you can still read these folders using ``load_waveforms``, which creates a ``MockWaveformExtractor`` object.
This object contains a ``SortingAnalyzer`` which can be accessed and then saved as follows

.. code-block:: python

    waveform_folder_path = "path_to/my_waveform_extractor_folder"
    new_sorting_analyzer_path = "path_to/my_new_sorting_analyzer_folder"
    # On Windows
    # new_sorting_analyzer_path = r"path_to\my_new_sorting_analyzer_folder"
    wvf_extractor = load_waveform(folder=waveform_folder_path)
    sorting_analyzer = wvf_extractor.sorting_analyzer
    sorting_analyzer.save_as(folder=new_sorting_analyzer_path, format="binary_folder")


The above code creates a ``SortingAnalyzer`` folder at ``new_sorting_analyzer_path``.

Dictionary between SortingAnalyzer and WaveformExtractor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provides a dictionary for code, to help translate between a ``SortingAnalyzer``
and a ``WaveformExtractor``, so that users can easily update old code. If you want to learn
how to use ``SortingAnalyzer`` from scratch, the
`Get Started <https://spikeinterface.readthedocs.io/en/latest/how_to/get_started.html>`_ guide
and the `postprocessing module documentation <https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html>`_
are better places to start.

This section is split into four subsections:

* `Create, load and save <#id2>`_
* `Checking basic properties <#id3>`_
* `Compute Extensions <#id4>`_
* `Quality Metrics <#id5>`_

Throughout this section, we assume that all functions have been imported into the namespace.
E.g. we have run code such as ``from spikeinterface.core import create_sorting_analyzer`` for each function. If you have imported
the full package (ie you have run ``import spikeinterface.full as si``) you need to prepend all
functions by ``si.``. If you have imported individual modules (ie you have run ``import spikeinterface.postprocessing as spost``)
we will need to prepend functions by the appropriate submodule name.


In the following we start with a recording called `recording` and a sorting
object called ``sorting``. We’ll then create, export, explore and compute things using a
``SortingAnalyzer`` called ``analyzer`` and a ``WaveformExtractor`` called ``wvf_extractor``.
We’ll do the same calculations for both objects. The WaveformExtractor code will be on
the left while the SortingAnalyzer code will be displayed on the right:

.. grid:: 2

    .. grid-item::

        **WaveformExtractor**

    .. grid-item::

        **SortingAnalyzer**

Create, load and save
+++++++++++++++++++++

First, create the object from a recording and a sorting.

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor = extract_waveforms(
                sorting=sorting,
                recording=recording
            )

    .. grid-item::

        .. code-block:: python

            analyzer = create_sorting_analyzer(
                sorting=sorting,
                recording=recording
            )

By default, the object is stored in memory. In this case, if you end your session without saving (which you can do using ``save_as``, see below) you'll lose everything!
Alternatively, we can save it locally at the point of creation by specifying a ``folder`` and a ``format``. Additionally, you can decide whether to use sparsity or not

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor = extract_waveforms(
                sorting=sorting,
                recording=recording,
                folder="my_waveform_extractor",
                mode="folder",
                sparse=True
            )

    .. grid-item::

        .. code-block:: python

            analyzer = create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                folder="my_sorting_analyzer",
                format="binary_folder",
                sparse=True
            )

You can save the object after you've created it, with the option
of saving it to a new format

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.save(
                format="zarr",
                folder="/path/to_my/result.zarr"
            )

    .. grid-item::

        .. code-block:: python

            analyzer.save_as(
                format="zarr",
                folder="/path/to_my/result.zarr"
            )




If you already have the object saved, you can load it

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor = load_waveforms(
                folder="my_waveform_extractor"
            )

    .. grid-item::

        .. code-block:: python

            analyzer = load_sorting_analyzer(
                folder="my_sorting_analyzer"
            )

Checking basic properties
+++++++++++++++++++++++++

The object contains both a ``sorting`` and a ``recording`` object. These
can be isolated


.. grid:: 2

    .. grid-item::

        .. code-block:: python

            the_recording = wvf_extractor.recording
            the_sorting = wvf_extractor.sorting

    .. grid-item::

        .. code-block:: python

            the_recording = analyzer.recording
            the_sorting = analyzer.sorting




You can then check any ``recording`` or ``sorting`` properties from these objects.

There is much information about the recording and sorting contained in the parent object. E.g. you can get
the channel locations as follows

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            channel_locations =
                wvf_extractor.get_channel_locations()

    .. grid-item::

        .. code-block:: python

            channel_locations =
                analyzer.get_channel_locations()




Many properties can be accessed in a similar way

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.get_num_channels()
            wvf_extractor.get_num_samples()
            wvf_extractor.get_num_segments()
            wvf_extractor.get_probe()
            wvf_extractor.get_probegroup()
            wvf_extractor.get_total_duration()
            wvf_extractor.get_total_samples()

    .. grid-item::

        .. code-block:: python

            analyzer.get_num_channels()
            analyzer.get_num_samples()
            analyzer.get_num_segments()
            analyzer.get_probe()
            analyzer.get_probegroup()
            analyzer.get_total_duration()
            analyzer.get_total_samples()

...while some are simply properties of the object

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.channel_ids
            wvf_extractor.unit_ids
            wvf_extractor.sampling_frequency

    .. grid-item::

        .. code-block:: python

            analyzer.channel_ids
            analyzer.unit_ids
            analyzer.sampling_frequency




You can also find some fundamental properties of the object,
though these are mostly used internally:

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.folder
            wvf_extractor.format
            wvf_extractor.is_read_only()
            wvf_extractor.dtype
            wvf_extractor.is_sparse()

    .. grid-item::

        .. code-block:: python

            analyzer.folder
            analyzer.format
            analyzer.is_read_only()
            analyzer.get_dtype()
            analyzer.is_sparse()

Compute Extensions
++++++++++++++++++

Waveforms, templates, quality metrics etc are all extensions of the ``SortingAnalyzer`` object.
Some extensions depend on other extensions. To calculate a *child* we must first have calculated its
**parents**. The relationship between some commonly used extensions are shown below:

.. image:: waveform_extractor_to_sorting_analyzer_files/child_parent_plot.svg
    :alt: Child parent relationships

We see that to compute ``spike_amplitudes`` we must first compute ``templates``. To compute templates
we must first compute ``waveforms``. To compute waveforms we must first compute ``random_spikes``. Phew!
Some of these extensions were calculated automatically for WaveformExtractors, so the code
looks slightly different. Let's calculate these extensions, and also add a parameter for ``spike_amplitudes``

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.precompute_templates(
                modes=("average",)
            )
            compute_spike_amplitudes(
                waveform_extractor=wvf_extractor,
                peak_sign="pos"
            )

    .. grid-item::

        .. code-block:: python

            analyzer.compute("random_spikes")
            analyzer.compute("waveforms")
            analyzer.compute("templates")
            analyzer.compute(
                "spike_amplitudes",
                peak_sign="pos"
            )

Read more about extensions and their keyword arguments in the
`postprocessing module documentation <https://spikeinterface.readthedocs.io/en/latest/modules/postprocessing.html>`_

In many cases, you can still use the old notation for ``SortingAnalyzer`` objects,
such as ``compute_spike_amplitudes(sorting_analyzer=analyzer)``.

In all cases, if the object has been saved locally, the extensions will be saved
locally too. You can check which extensions have been saved

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.get_available_extension_names()

    .. grid-item::

        .. code-block:: python

            analyzer.get_saved_extension_names()

You can now also check which extensions are currently loaded *in memory*. The WaveformExtractor
checks the local folder *and* the memory:

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.get_available_extension_names()

    .. grid-item::

        .. code-block:: python

            analyzer.get_loaded_extension_names()

If there is an extensions which is saved but not yet loaded you can load it:

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.load_extension(
                extension_name="spike_amplitudes"
            )

    .. grid-item::

        .. code-block:: python

            analyzer.load_extension(
                extension_name="spike_amplitudes"
            )

You can also check if a certain extension is loaded

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            wvf_extractor.has_extension(
                extension_name="spike_amplitudes"
            )


    .. grid-item::

        .. code-block:: python

            analyzer.has_extension(
                extension_name="spike_amplitudes"
            )

You can delete extensions. Note that if you delete a parent, all of its children
will be deleted too. We'll now delete ``templates`` from the SortingAnalyzer and ``spike_amplitudes`` from our WaveformExtractor.

.. grid:: 2

    .. grid-item::

          .. code-block:: python

            wvf_extractor.delete_extension(
                extension_name="spike_amplitudes"
            )

    .. grid-item::

        .. code-block:: python

            # This also deletes any children
            # such as spike_amplitudes
            analyzer.delete_extension(
                extension_name="templates"
            )

Once you have computed an extension, you often want to look at the data associated with it.
This has been standardized for the ``SortingAnalyzer`` object, through the ``get_data`` method.
The retrieval methods for the ``WaveformExtractor`` object were less uniform, and depended
on which extension you were interested in. We won't list them all here.

.. grid:: 2

    .. grid-item::

          .. code-block:: python

            wv_data = wvf_extractor.get_waveforms(
                unit_id=0
            )

            ul_data = compute_unit_locations(
                waveform_extractor=wvf_extractor
            )

    .. grid-item::

        .. code-block:: python

            wv = analyzer.get_extension(
                extension_name="waveforms"
            )
            wv_data = wv.get_data()
            ul = analyzer.get_extension(
                extension_name="unit_locations"
            )
            ul_data = ul.get_data()

You can also access the parameters used in the extension calculation, which is very simple for the new SortingAnalyzer:

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            ul_ex = wvf_extractor.load_extension(
                extension_name="unit_locations"
            )
            ul_parms = ul_ex.load_params_from_folder(
                folder="my_waveform_extractor"
            )

    .. grid-item::

        .. code-block:: python

            ul_params = ul.params

Quality metrics
+++++++++++++++

Quality metrics for the ``SortingAnalyzer`` are also extensions. You can calculate a specific
quality metric using the ``metric_names`` argument. In contrast, for WaveformExtractors  you
need to find the correct function. The old functions still work for SortingAnalyzers.

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            amp_cut_data = compute_amplitude_cutoffs(
                waveform_extractor=wvf_extractor
            )
            #or: compute_amplitude_cutoffs(
            #        wvf_extractor
            #    )

    .. grid-item::

        .. code-block:: python

            amp_cutoff = analyzer.compute(
                "quality_metrics",
                metric_names=["amplitude_cutoff"]
            )
            amp_cut_data = amp_cutoff.get_data()
            #or: compute_amplitude_cutoff(analyzer)

Or you can calculate all available quality metrics. Here, we also pass a
list of quality metric parameters.

.. grid:: 2

    .. grid-item::

        .. code-block:: python

            dqm_params = get_default_qm_params()
            amp_cut_data = compute_quality_metrics(
                waveform_extractor=wvf_extractor,
                qm_params=dqm_params
            )

    .. grid-item::

        .. code-block:: python

            dqm_params = get_default_qm_params()
            amp_cutoff = analyzer.compute(
                "quality_metrics",
                qm_params=dqm_params
            )
            #alt: compute_quality_metrics(analyzer)

Learn more about the possible quality metrics and their keyword arguments in the
`quality metrics documentation page <https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html>`_.
