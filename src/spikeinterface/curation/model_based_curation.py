import numpy as np
from pathlib import Path
import json
import pandas as pd
import warnings

from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation.train_manual_curation import try_to_get_metrics_from_analyzer


class ModelBasedClassification:
    """
    Class for performing model-based classification on spike sorting data.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting data.
    pipeline : Pipeline
        The pipeline object representing the trained classification model.

    Attributes
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting data.
    pipeline : Pipeline
        The pipeline object representing the trained classification model.
    required_metrics : Sequence[str]
        The list of required metrics for classification, extracted from the pipeline.

    Methods
    -------
    predict_labels()
        Predicts the labels for the spike sorting data using the trained model.
    _get_metrics_for_classification()
        Retrieves the metrics data required for classification.
    _check_params_for_classification()
        Checks if the parameters for classification match the training parameters.
    """

    def __init__(self, sorting_analyzer: SortingAnalyzer, pipeline):
        from sklearn.pipeline import Pipeline

        if not isinstance(pipeline, Pipeline):
            raise ValueError("The pipeline must be an instance of sklearn.pipeline.Pipeline")

        self.sorting_analyzer = sorting_analyzer
        self.pipeline = pipeline
        self.required_metrics = pipeline.feature_names_in_

    def predict_labels(self, label_conversion=None, input_data=None, export_to_phy=False, pipeline_info_path=None):
        """
        Predicts the labels for the spike sorting data using the trained model.
        Populates the sorting object with the predicted labels and probabilities as unit properties

        Parameters
        ----------
        input_data : pandas.DataFrame, optional
            The input data for classification. If not provided, the method will extract metrics stored in the sorting analyzer.
        export_to_phy : bool, optional
            Whether to export the classified units to Phy format. Default is False.
        pipeline_info_path : str or Path, default: None
            Path to pipeline info, used to check metric parameters used to train Pipeline.

        Returns
        -------
        dict
            A dictionary containing the classified units and their corresponding predictions and probabilities.
            The dictionary has the format {unit_id: (prediction, probability)}.
        """

        # Get metrics DataFrame for classification
        if input_data is None:
            input_data = self._get_metrics_for_classification()
        else:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")

        # Check all the required metrics have been calculated
        calculated_metrics = set(input_data.keys())
        required_metrics = set(self.required_metrics)
        if required_metrics.issubset(calculated_metrics):
            input_data = input_data[self.required_metrics]
        else:
            print("Input data does not contain all required metrics for classification")
            print("Missing metrics: ", required_metrics.difference(calculated_metrics))

        print(pipeline_info_path)
        self._check_params_for_classification(pipeline_info_path=pipeline_info_path)

        # Prepare input data
        input_data = input_data.map(lambda x: np.nan if np.isinf(x) else x)
        input_data = input_data.astype("float32")

        # Apply classifier
        predictions = self.pipeline.predict(input_data)
        probabilities = self.pipeline.predict_proba(input_data)
        probabilities = np.max(probabilities, axis=1)

        if isinstance(label_conversion, dict):
            try:
                assert set(predictions).issubset(label_conversion.keys())
            except AssertionError:
                raise ValueError("Labels in predictions do not match those in label_conversion")
            predictions = [label_conversion[label] for label in predictions]

        # Make output dict with {unit_id: (prediction, probability)}
        classified_units = {
            unit_id: (prediction, probability)
            for unit_id, prediction, probability in zip(input_data.index, predictions, probabilities)
        }

        # Set predictions and probability as sorting properties
        self.sorting_analyzer.sorting.set_property("label_prediction", predictions)
        self.sorting_analyzer.sorting.set_property("label_confidence", probabilities)

        if export_to_phy:
            self._export_to_phy(classified_units)

        return classified_units

    def _get_metrics_for_classification(self):
        """Check if all required metrics are present and return a DataFrame of metrics for classification"""

        quality_metrics, template_metrics = try_to_get_metrics_from_analyzer(self.sorting_analyzer)

        # Create DataFrame of all metrics and reorder columns to match the model
        calculated_metrics = pd.concat([quality_metrics, template_metrics], axis=1)

        # Remove any metrics for non-existent units, raise error if no units are present
        calculated_metrics = calculated_metrics.loc[
            calculated_metrics.index.isin(self.sorting_analyzer.sorting.get_unit_ids())
        ]
        if calculated_metrics.shape[0] == 0:
            raise ValueError("No units present in sorting data")

        return calculated_metrics

    def _check_params_for_classification(self, pipeline_info_path):
        """
        Check that quality and template metrics parameters match those used to train the model

        Parameters
        ----------
        pipeline_info_path : str or Path
            Path to pipeline_info.json provenance file
        """

        quality_metrics_extension = self.sorting_analyzer.get_extension("quality_metrics")
        template_metrics_extension = self.sorting_analyzer.get_extension("template_metrics")

        if pipeline_info_path is None:
            try:
                pipeline_info_path = self.pipeline.output_folder + "/model_info.json"
            except:
                pass
        else:
            try:
                with open(pipeline_info_path) as fd:
                    pipeline_info = json.load(fd)
            except:
                warnings.warn(
                    "Could not check if metric parameters are consistent between trained model and this sorting_analyzer. Please supply the path to the `model_info.json` file using `pipeline_info_path` to check this."
                )
                return

        if quality_metrics_extension is not None:

            pipeline_quality_metrics_params = pipeline_info["metric_params"]["analyzer_0"]["quality_metric_params"][
                "qm_params"
            ]
            quality_metrics_params = quality_metrics_extension.params["qm_params"]

            # need to make sure both dicts are in json format, so that lists are equal
            if json.dumps(quality_metrics_params) != json.dumps(pipeline_quality_metrics_params):
                warnings.warn(
                    "Quality metrics params do not match those used to train pipeline. Check these in the 'pipeline_info.json' file."
                )

        if template_metrics_extension is not None:

            pipeline_template_metrics_params = pipeline_info["metric_params"]["analyzer_0"]["template_metric_params"][
                "metrics_kwargs"
            ]
            template_metrics_params = template_metrics_extension.params["metrics_kwargs"]

            if template_metrics_params != pipeline_template_metrics_params:
                warnings.warn(
                    "Template metrics metrics params do not match those used to train pipeline. Check these in the 'model_info.json' file."
                )

    def _export_to_phy(self, classified_units):
        """Export the classified units to Phy as cluster_prediction.tsv file"""

        # Create a new DataFrame with unit_id, prediction, and probability columns from dict {unit_id: (prediction, probability)}
        classified_df = pd.DataFrame.from_dict(classified_units, orient="index", columns=["prediction", "probability"])

        # Export to Phy format
        try:
            sorting_path = self.sorting_analyzer.sorting.get_annotation("phy_folder")
            assert sorting_path is not None
            assert Path(sorting_path).is_dir()
        except AssertionError:
            raise ValueError("Phy folder not found in sorting annotations, or is not a directory")

        classified_df.to_csv(f"{sorting_path}/cluster_prediction.tsv", sep="\t", index_label="cluster_id")


def auto_label_units(
    sorting_analyzer: SortingAnalyzer,
    pipeline,
    label_conversion=None,
    export_to_phy=False,
    pipeline_info_path=None,
):
    """
    Automatically labels units based on a model-based classification.

    This function populates the sorting object with the predicted labels and probabilities as unit properties.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting results.
    pipeline : Pipeline
        The scikit-learn pipeline object containing the model-based classification pipeline.
    label_conversion : dict, optional
        A dictionary for converting the predicted labels (which are integers) to custom labels. The dictionary should have the format {old_label: new_label}.
    export_to_phy : bool, optional
        Whether to export the results to Phy format. Default is False.
    pipeline_info_path : str or Path, default: None
        Path to pipeline info, used to check metric parameters used to train Pipeline.

    Returns
    -------
    classified_units : dict
        A dictionary containing the classified units, where the keys are the unit IDs and the values are a tuple of labels and confidence.

    Raises
    ------
    ValueError
        If the pipeline is not an instance of sklearn.pipeline.Pipeline.

    """
    from sklearn.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise ValueError("The pipeline must be an instance of sklearn.pipeline.Pipeline")

    model_based_classification = ModelBasedClassification(sorting_analyzer, pipeline)

    classified_units = model_based_classification.predict_labels(
        label_conversion=label_conversion, export_to_phy=export_to_phy, pipeline_info_path=pipeline_info_path
    )

    return classified_units
