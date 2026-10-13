Use a trained model to predict the curation labels
==================================================

For a more detailed guide to using trained models, `read our tutorial here
<https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_1_automated_curation.html>`_).

There is a Collection of models for automated curation available on the
`SpikeInterface HuggingFace page <https://huggingface.co/SpikeInterface>`_.

We'll apply the model ``toy_tetrode_model`` from ``SpikeInterface`` on a SortingAnalyzer
called ``sorting_analyzer``. We assume that the quality and template metrics have
already been computed.

We need to pass the ``sorting_analyzer``, the ``repo_id`` (which is just the part of the
repo's URL after huggingface.co/) and that we trust the model.

.. code::

    from spikeinterface.curation import model_based_label_units

    labels_and_probabilities = model_based_label_units(
        sorting_analyzer = sorting_analyzer,
        repo_id = "SpikeInterface/toy_tetrode_model",
        trust_model = True
    )

If you have a local directory containing the model in a ``skops`` file you can use this to
create the labels:

.. code::

    labels_and_probabilities = si.model_based_label_units(
        sorting_analyzer = sorting_analyzer,
        model_folder = "my_folder_with_a_model_in_it",
    )

The returned labels are a dictionary of model's predictions and it's confidence. These
are also saved as a property of your ``sorting_analyzer`` and can be accessed like so:

.. code::

    labels = sorting_analyzer.get_sorting_property("classifier_label")
    probabilities = sorting_analyzer.get_sorting_property("classifier_probability")
