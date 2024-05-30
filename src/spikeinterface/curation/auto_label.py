from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from spikeinterface.core import SortingAnalyzer

class ModelBasedClassification:
    # TODO docstring
    
    def __init__(self, sorting_analyzer:SortingAnalyzer, pipeline:Pipeline, required_metrics:Sequence[str]):
        
        self.sorting_analyzer = sorting_analyzer
        
        self.required_metrics = required_metrics
        self.pipeline = pipeline

    def predict_labels(self):
        # Get metrics DataFrame for classification
        input_data = self._get_metrics_for_classification()

        # Prepare input data
        input_data[np.isinf(input_data)] = np.nan
        input_data = input_data.astype('float32')

        # Apply classifier
        predictions = self.pipeline.predict(input_data)
        probabilities = self.pipeline.predict_proba(input_data)

        # Make output dict with {unit_id: (prediction, probability)}
        classified_units = {unit_id: (prediction, probability) for unit_id, prediction, probability in zip(input_data.index, predictions, probabilities)}

        return classified_units

    def _get_metrics_for_classification(self):
                                 
        try:
            quality_metrics = self.sorting_analyzer.extensions['quality_metrics'].data["metrics"]
            template_metrics = self.sorting_analyzer.extensions['template_metrics'].data["metrics"]
        except KeyError:
            raise ValueError("Quality and template metrics must be computed before classification")
        
        # Check if any metrics are missing
        metrics_list = quality_metrics.columns.to_list() + template_metrics.columns.to_list()
        missing_metrics = [metric for metric in self.required_metrics if metric not in metrics_list]
        
        if len(missing_metrics) > 0:
            raise ValueError(f"Missing metrics: {missing_metrics}")
        
        # Create DataFrame of all metrics and reorder columns to match the model
        calculated_metrics = pd.concat([quality_metrics, template_metrics], axis = 1)
        calculated_metrics = calculated_metrics[self.required_metrics]

        return calculated_metrics
    
def auto_label_units(sorting_analyzer:SortingAnalyzer, pipeline:Pipeline, required_metrics:Sequence[str]):
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
