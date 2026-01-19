Train a model to predict curation labels
========================================

A full tutorial for model-based curation can be found `here <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_2_train_a_model.html>`_.

Here, we assume that you have:

* Two SortingAnalyzers called ``analyzer_1`` and
  ``analyzer_2``, and have calculated some template and quality metrics for both
* Manually curated labels for the units in each analyzer, in lists called
  ``analyzer_1_labels`` and ``analyzer_2_labels``. If you have used phy, the lists can
  be accessed using ``curated_labels = analyzer.sorting.get_property("quality")``.

With these objects calculated, you can train a model as follows

.. code::

    from spikeinterface.curation import train_model

    analyzer_list = [analyzer_1, analyzer_2]
    labels_list = [analyzer_1_labels, analyzer_2_labels]
    output_folder = "/path/to/output_folder"

    trainer = train_model(
        mode="analyzers",
        labels=labels_list,
        analyzers=analyzer_list,
        output_folder=output_folder,
        metric_names=None, # Set if you want to use a subset of metrics, defaults to all calculated quality and template metrics
        imputation_strategies=None, # Default is all available imputation strategies
        scaling_techniques=None, # Default is all available scaling techniques
        classifiers=None, # Defaults to Random Forest classifier only - we usually find this gives the best results, but a range of classifiers is available
        seed=None, # Set a seed for reproducibility
    )


The trainer tries several models and chooses the most accurate one. This model and
some metadata are stored in the ``output_folder``, which can later be loaded using the
``load_model`` function (`more details <https://spikeinterface.readthedocs.io/en/latest/tutorials/curation/plot_1_automated_curation.html#download-a-pretrained-model>`_).
We can also access the model, which is an sklearn ``Pipeline``, from the trainer object

.. code::

    best_model = trainer.best_pipeline


The training function can also be run in “csv” mode, if you prefer to
store metrics in as .csv files. If the target labels are stored as a column in
the file, you can point to these with the ``target_label`` parameter

.. code::

    trainer = train_model(
        mode="csv",
        metrics_paths = ["/path/to/csv_file_1", "/path/to/csv_file_2"],
        target_label = "my_label",
        output_folder=output_folder,
    )
