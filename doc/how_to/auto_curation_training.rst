How to train a model to predict curation labels
===============================================

-  This assumes you alrady have your data loaded as one or several
   SortingAnalyzers, and have calculated some quality metrics -
   `tutorial
   here <https://spikeinterface.readthedocs.io/en/latest/tutorials/qualitymetrics/plot_3_quality_mertics.html>`
-  You also need a list of cluster labels for each SortingAnalyzer,
   which can be extracted from a SortingAnalyzer, from phy, or loaded
   from elsewhere
-  Full tutorial for model-based curation can be found
   `here<https://spikeinterface.readthedocs.io/en/latest/tutorials/qualitymetrics/plot_5_automated_curation.html>`

.. code::

    from pathlib import Path
    import pandas as pd
    from spikeinterface.curation.train_manual_curation import train_model

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


    best_model = trainer.best_pipeline
    best_model

Load and disply top 5 pipelines and accuracies

.. code::

    accuracies = pd.read_csv(Path(output_folder) / Path("model_label_accuracies.csv"), index_col = 0)
    accuracies.head()

This training function can also be run in “csv” mode if you want to
store metrics in a single .csv file. If the target labels are stored in
the file, you can point to these with the ``target_label`` parameter

.. code::

    trainer = train_model(
        mode="csv",
        metrics_path = "/path/to/csv",
        target_label = "label",
        output_folder=output_folder,
        metric_names=None, # Set if you want to use a subset of metrics, defaults to all calculated quality and template metrics
        imputation_strategies=None, # Default is all available imputation strategies
        scaling_techniques=None, # Default is all available scaling techniques
        classifiers=None, # Defaults to Random Forest classifier only - we usually find this gives the best results, but a range of classifiers is available
        seed=None, # Set a seed for reproducibility
    )
