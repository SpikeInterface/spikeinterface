from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from spikeinterface.core import SortingAnalyzer
from spikeinterface.qualitymetrics.quality_metric_calculator import get_default_qm_params
from spikeinterface.postprocessing.template_metrics import _default_function_kwargs as default_template_metrics_params


class ModelBasedClassification:
    """
    Class for performing model-based classification on spike sorting data.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting data.
    pipeline : Pipeline
        The pipeline object representing the trained classification model.
    required_metrics : Sequence[str]
        The list of required metrics for classification.

    Attributes
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting data.
    required_metrics : Sequence[str]
        The list of required metrics for classification.
    pipeline : Pipeline
        The pipeline object representing the trained classification model.

    Methods
    -------
    predict_labels()
        Predicts the labels for the spike sorting data using the trained model.
    _get_metrics_for_classification()
        Retrieves the metrics data required for classification.
    _check_params_for_classification()
        Checks if the parameters for classification match the training parameters.
    """

    def __init__(self, sorting_analyzer: SortingAnalyzer, pipeline: Pipeline, required_metrics: Sequence[str]):

        self.sorting_analyzer = sorting_analyzer
        self.required_metrics = required_metrics
        self.pipeline = pipeline

    def predict_labels(self):
        """
        Predicts the labels for the spike sorting data using the trained model.

        Returns
        -------
        dict
            A dictionary containing the classified units and their corresponding predictions and probabilities.
            The dictionary has the format {unit_id: (prediction, probability)}.
        """

        # Get metrics DataFrame for classification
        input_data = self._get_metrics_for_classification()

        # Check params match training data
        self._check_params_for_classification()

        # Prepare input data
        input_data = input_data.map(lambda x: np.nan if np.isinf(x) else x)
        input_data = input_data.astype("float32")

        print(input_data)

        # Apply classifier
        predictions = self.pipeline.predict(input_data)
        probabilities = self.pipeline.predict_proba(input_data)

        # Make output dict with {unit_id: (prediction, probability)}
        classified_units = {
            unit_id: (prediction, probability)
            for unit_id, prediction, probability in zip(input_data.index, predictions, probabilities)
        }

        return classified_units

    def _get_metrics_for_classification(self):
        """Check if all required metrics are present and return a DataFrame of metrics for classification"""

        try:
            quality_metrics = self.sorting_analyzer.extensions["quality_metrics"].data["metrics"]
            template_metrics = self.sorting_analyzer.extensions["template_metrics"].data["metrics"]
        except KeyError:
            raise ValueError("Quality and template metrics must be computed before classification")

        # Check if any metrics are missing
        metrics_list = quality_metrics.columns.to_list() + template_metrics.columns.to_list()
        missing_metrics = [metric for metric in self.required_metrics if metric not in metrics_list]

        if len(missing_metrics) > 0:
            raise ValueError(f"Missing metrics: {missing_metrics}")

        # Create DataFrame of all metrics and reorder columns to match the model
        calculated_metrics = pd.concat([quality_metrics, template_metrics], axis=1)
        calculated_metrics = calculated_metrics[self.required_metrics]

        return calculated_metrics

    def _check_params_for_classification(self):
        """Check that quality and template metrics parameters match those used to train the model
        NEEDS UPDATING TO PULL IN PARAMS FROM TRAINING DATA"""

        try:
            quality_metrics_params = self.sorting_analyzer.extensions["quality_metrics"].params["qm_params"]
            template_metric_params = self.sorting_analyzer.extensions["template_metrics"].params["metrics_kwargs"]
        except KeyError:
            raise ValueError("Quality and template metrics must be computed before classification")

        # TODO: check metrics_params match those used to train the model - how?
        # TEMP - check that params match the default. Need to add ability to check against model training params
        default_quality_metrics_params = get_default_qm_params()
        default_template_metrics_params

        # Check that dicts are identical
        if quality_metrics_params != default_quality_metrics_params:
            raise ValueError("Quality metrics params do not match default params")
        elif template_metric_params != default_template_metrics_params:
            raise ValueError("Template metrics params do not match default params")
        else:
            pass

        # TODO: decide whether to also check params against parent extensions of metrics (e.g. waveforms, templates)
        # This would need to account for the fact that these extensions may no longer exist


def auto_label_units(sorting_analyzer: SortingAnalyzer, pipeline: Pipeline, required_metrics: Sequence[str]):
    """
    Automatically labels units based on a model-based classification.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting results.
    pipeline : Pipeline
        The pipeline object containing the model-based classification pipeline.
    required_metrics : Sequence[str]
        The list of required metrics used for classification.

    Returns
    -------
    classified_units : dict
        A dictionary containing the classified units, where the keys are the unit IDs and the values are a tuple of labels and confidence.

    """
    model_based_classification = ModelBasedClassification(sorting_analyzer, pipeline, required_metrics)

    classified_units = model_based_classification.predict_labels()

    return classified_units
