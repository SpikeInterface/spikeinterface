Postprocessing module
=====================


After spike sorting, we can use the :py:mod:`~spikeinterface.postprocessing` module to further post-process
the spike sorting output. Most of the post-processing functions require a
:py:class:`~spikeinterface.core.WaveformExtractor` as input. There are several postprocessing tools available, and all 
of them are implemented as a :py:class:`~spikeinterface.core.BaseWaveformExtractorExtension`. These objects are tightly 
connected to the parent :code:`WaveformExtractor` object, so that operations done on the :code:`WaveformExtractor`,
such as saving, loading, or selecting units, will be automatically applied to all extensions.

To check what extensions are available for a :code:`WaveformExtractor` named :code:`we`, you can use:

.. code-block:: python

    import spikeinterface as si

    available_extension_names = we.get_available_extension_names()
    print(available_extension_names)

.. code-block:: bash

    ["principal_components", "spike_amplitudes"]

In this case, for example, principal components and spike amplitudes have been already computed.
To load the extension object you can run:

.. code-block:: python

    ext = we.load_extension("spike_amplitudes")
    ext_data = ext.get_data()

Here :code:`ext` is the extension object (in this case the :code:`SpikeAmplitudeCalculator`), and :code:`ext_data` will 
contain the actual amplitude data. Note that different extensions might have different ways to return the extension.
You can use :code:`ext.get_data?` for documentation.



Available postprocessing extensions in SpikeInterface are are:

principal_components
--------------------

This extension computes the principal components of the waveforms. There are several modes available:

* "by_channel_local" (default): fits one PCA model for each by_channel
* "by_channel_global": fits the same PCA model to all channels (also termed temporal PCA)
* "concatenated": contatenates all channels and fits a PCA model on the concatenated data

If the input :code:`WaveformExtractor` is sparse, the sparsity is used when computing PCA.
For dense waveforms, sparsity can also be passed as an argument.

Reference
^^^^^^^^^

.. automodule:: spikeinterface.qualitymetrics.postprocessing

	.. autofunction:: compute_principal_components

template_similarity
--------------------

This extension computes the similarity of the templates to each other. This information could be used for automatic 
merging. Currently, the only available similarity method is the cosine similarity, which is the angle between the 
high-dimensional flattened template arrays. Note that cosine similarity does not take into account amplitude differences 
and is not well suited for high-density probes.


spike_amplitudes
----------------

This extension computes the amplitude of each spike as the value of the traces on the extremum channel at the times of 
each spike.

Reference
^^^^^^^^^

.. automodule:: spikeinterface.qualitymetrics.postprocessing

	.. autofunction:: compute_spike_amplitudes

Reference
^^^^^^^^^

.. automodule:: spikeinterface.qualitymetrics.postprocessing

	.. autofunction:: compute_template_similarity

template_metrics
----------------

TODO

spike_amplitudes
----------------

This extension computes the similarity of the templates to each other. This information could be used for automatic 
merging. Currently, the only available similarity method is the cosine similarity, which is the angle between the 
high-dimensional flattened template arrays. Note that cosine similarity does not take into account amplitude differences 
and is not well suited for high-density probes.

Reference
^^^^^^^^^

.. automodule:: spikeinterface.qualitymetrics.postprocessing

	.. autofunction:: compute_template_similarity

correlograms 
-------------

TODO

ISI
---

TODO

noise level
-----------

TODO


spike locations
---------------

TODO

template metrics
----------------

TODO


unit localisation
-----------------

TODO


template metrics
----------------

TODO


align sorting
-------------

TODO
