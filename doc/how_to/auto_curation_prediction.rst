How to use a trained model to predict the curation labels
=========================================================

This assumes you alrady have your data loaded as a SortingAnalyzer, and have calculated some quality metrics - `tutorial here <https://spikeinterface.readthedocs.io/en/latest/tutorials/qualitymetrics/plot_3_quality_metrics.html>`_
Full tutorial for model-based curation can be found `here <https://spikeinterface.readthedocs.io/en/latest/tutorials/qualitymetrics/plot_5_automated_curation.html>`_
    - Pre-trained models can be downloaded from `Hugging Face <https://huggingface.co/>`_, or opened from `skops <https://skops.readthedocs.io/en/stable/>`_ files
    - The model (``.skops``) file AND the ``pipeline_info.json`` (both produced when training the model) are required for full prediction


.. code:: 

    # Load a model
    from huggingface_hub import hf_hub_download
    import skops.io
    import json

    # Download the model and json file from Hugging Face (can also load from local paths)
    repo = "chrishalcrow/test_automated_curation_3"
    model_path = hf_hub_download(repo_id=repo, filename="best_model_label.skops")
    json_path = hf_hub_download(repo_id=repo, filename="model_pipeline.json")
    model = skops.io.load(model_path, trusted='numpy.dtype')
    pipeline_info = json.load(open(json_path))

Use the model to predict labels on your SortingAnalyzer

.. code:: 

    from spikeinterface.curation.model_based_curation import auto_label_units

    analyzer # Your SortingAnalyzer

    label_conversion = pipeline_info['label_conversion']
    label_dict = auto_label_units(sorting_analyzer=analyzer,
                                  pipeline=model,
                                  label_conversion=label_conversion,
                                  export_to_phy=False,
                                  pipeline_info_path=None)

    # The labels are stored in the sorting "label_predictions" and "label_confidence" property
    analyzer.sorting
