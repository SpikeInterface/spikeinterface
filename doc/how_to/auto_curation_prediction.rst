How to use a trained model to predict the curation labels
=========================================================

There is a Collection of models for automated curation available on the
`SpikeInterface HuggingFace page <https://huggingface.co/SpikeInterface>`_.

We'll apply the model ``toy_tetrode_model`` from ``SpikeInterface`` on a SortingAnalyzer
called ``sorting_analyzer``. We assume that the quality and template metrics have
already been computed (`full tutorial here <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_1_automated_curation.html>`_).

We need to pass the ``sorting_analyzer``, the ``repo_id`` (which is just the part of the
repo's URL after huggingface.co/) and the trusted types list (see more `here <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_1_automated_curation.html#a-more-realistic-example>`_):

.. code::

    from spikeinterface.curation import auto_label_units

    labels_and_probabilities = auto_label_units(
        sorting_analyzer = sorting_analyzer,
        repo_id = "SpikeInterface/toy_tetrode_model",
        trusted = ['numpy.dtype']
    )

If you have a local directory containing the model in a ``skops`` file you can use this to
create the labels:

.. code::

    labels_and_probabilities = si.auto_label_units(
        sorting_analyzer = sorting_analyzer,
        model_folder = "my_folder_with_a_model_in_it",
    )

The returned labels are a dictionary of model's predictions and it's confidence. These
are also saved as a property of your ``sorting_analyzer`` and can be accessed like so:

.. code::

    labels = sorting_analyzer.sorting.get_property("classifier_label")
    probabilities = sorting_analyzer.sorting.get_property("classifier_probability")
