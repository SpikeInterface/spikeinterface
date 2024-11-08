import numpy as np
from pathlib import Path
import json
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
    """

    def __init__(self, sorting_analyzer: SortingAnalyzer, pipeline):
        from sklearn.pipeline import Pipeline

        if not isinstance(pipeline, Pipeline):
            raise ValueError("The `pipeline` must be an instance of sklearn.pipeline.Pipeline")

        self.sorting_analyzer = sorting_analyzer
        self.pipeline = pipeline
        self.required_metrics = pipeline.feature_names_in_

    def predict_labels(
        self, label_conversion=None, input_data=None, export_to_phy=False, model_info=None, enforce_metric_params=False
    ):
        """
        Predicts the labels for the spike sorting data using the trained model.
        Populates the sorting object with the predicted labels and probabilities as unit properties

        Parameters
        ----------
        model_info : dict or None, default: None
            Model info, generated with model, used to check metric parameters used to train it.
        label_conversion : dict or None, default: None
            A dictionary for converting the predicted labels (which are integers) to custom labels. If None,
            tries to find in `model_info` file. The dictionary should have the format {old_label: new_label}.
        input_data : pandas.DataFrame or None, default: None
            The input data for classification. If not provided, the method will extract metrics stored in the sorting analyzer.
        export_to_phy : bool, default: False.
            Whether to export the classified units to Phy format. Default is False.
        enforce_metric_params : bool, default: False
            If True and the parameters used to compute the metrics in `sorting_analyzer` are different than the parmeters
            used to compute the metrics used to train the model, this function will raise an error. Otherwise, a warning is raised.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the classified units and their corresponding predictions and probabilities,
            indexed by their `unit_ids`.
        """
        import pandas as pd

        # Get metrics DataFrame for classification
        if input_data is None:
            input_data = self._get_metrics_for_classification()
        else:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")

        if model_info is not None:
            self._check_params_for_classification(enforce_metric_params, model_info=model_info)

        if model_info is not None and label_conversion is None:
            try:
                string_label_conversion = model_info["label_conversion"]
                # json keys are strings; we convert these to ints
                label_conversion = {}
                for key, value in string_label_conversion.items():
                    label_conversion[int(key)] = value
            except:
                warnings.warn("Could not find `label_conversion` key in `model_info.json` file")

        # Prepare input data
        input_data = input_data.applymap(lambda x: np.nan if np.isinf(x) else x)
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

        classified_units = pd.DataFrame(
            zip(predictions, probabilities), columns=["prediction", "probability"], index=self.sorting_analyzer.unit_ids
        )

        # Set predictions and probability as sorting properties
        self.sorting_analyzer.sorting.set_property("classifier_label", predictions)
        self.sorting_analyzer.sorting.set_property("classifier_probability", probabilities)

        if export_to_phy:
            self._export_to_phy(classified_units)

        return classified_units

    def _get_metrics_for_classification(self):
        """Check if all required metrics are present and return a DataFrame of metrics for classification"""

        import pandas as pd

        quality_metrics, template_metrics = try_to_get_metrics_from_analyzer(self.sorting_analyzer)

        # Create DataFrame of all metrics and reorder columns to match the model
        calculated_metrics = pd.concat([quality_metrics, template_metrics], axis=1)

        # Remove any metrics for non-existent units, raise error if no units are present
        calculated_metrics = calculated_metrics.loc[
            calculated_metrics.index.isin(self.sorting_analyzer.sorting.get_unit_ids())
        ]
        if calculated_metrics.shape[0] == 0:
            raise ValueError("No units present in sorting data")

        # Check all the required metrics have been calculated
        required_metrics = set(self.required_metrics)
        if required_metrics.issubset(set(calculated_metrics)):
            input_data = calculated_metrics[self.required_metrics]
        else:
            raise ValueError(
                "Input data does not contain all required metrics for classification",
                f"Missing metrics: {required_metrics.difference(calculated_metrics)}",
            )

        return input_data

    def _check_params_for_classification(self, enforce_metric_params=False, model_info=None):
        """
        Check that quality and template metrics parameters match those used to train the model

        Parameters
        ----------
        enforce_metric_params : bool, default: False
            If True and the parameters used to compute the metrics in `sorting_analyzer` are different than the parmeters
            used to compute the metrics used to train the model, this function will raise an error. Otherwise, a warning is raised.
        model_info_path : str or Path, default: None
            Path to model_info.json provenance file
        """

        quality_metrics_extension = self.sorting_analyzer.get_extension("quality_metrics")
        template_metrics_extension = self.sorting_analyzer.get_extension("template_metrics")

        if quality_metrics_extension is not None:

            model_quality_metrics_params = model_info["metric_params"]["quality_metric_params"]
            quality_metrics_params = quality_metrics_extension.params["qm_params"]

            inconsistent_metrics = []
            for metric in model_quality_metrics_params["metric_names"]:
                if metric not in model_quality_metrics_params["qm_params"]:
                    inconsistent_metrics += metric
                else:
                    if quality_metrics_params[metric] != model_quality_metrics_params["qm_params"][metric]:
                        warning_message = f"Quality metric params for {metric} do not match those used to train the model. Parameters can be found in the 'model_info.json' file."
                        if enforce_metric_params is True:
                            raise Exception(warning_message)
                        else:
                            warnings.warn(warning_message)

            if len(inconsistent_metrics) > 0:
                warning_message = (
                    f"Parameters used to compute metrics {inconsistent_metrics}, used to train this model, are unknown."
                )
                if enforce_metric_params is True:
                    raise Exception(warning_message)
                else:
                    warnings.warn(warning_message)

        if template_metrics_extension is not None:

            model_template_metrics_params = model_info["metric_params"]["template_metric_params"]["metrics_kwargs"]
            template_metrics_params = template_metrics_extension.params["metrics_kwargs"]

            if template_metrics_params == {}:
                warning_message = "Parameters used to compute template metrics, used to train this model, are unknown."
                if enforce_metric_params is True:
                    raise Exception(warning_message)
                else:
                    warnings.warn(warning_message)

            if template_metrics_params != model_template_metrics_params:
                warning_message = "Template metrics params do not match those used to train model. Parameters can be found in the 'model_info.json' file."
                if enforce_metric_params is True:
                    raise Exception(warning_message)
                else:
                    warnings.warn(warning_message)

    def _export_to_phy(self, classified_units):
        """Export the classified units to Phy as cluster_prediction.tsv file"""

        import pandas as pd

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
    model_folder=None,
    model_name=None,
    repo_id=None,
    label_conversion=None,
    trust_model=False,
    trusted=None,
    export_to_phy=False,
    enforce_metric_params=False,
):
    """
    Automatically labels units based on a model-based classification, either from a model
    hosted on HuggingFaceHub or one available in a local folder.

    This function returns the predicted labels and the prediction probabilities, and populates
    the sorting object with the predicted labels and probabilities in the 'classifier_label' and
    'classifier_probability' properties.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting results.
    model_folder : str or Path, defualt: None
        The path to the folder containing the model
    repo_id : str | Path, default: None
        Hugging face repo id which contains the model e.g. 'username/model'
    model_name: str | Path, default: None
        Filename of model e.g. 'my_model.skops'. If None, uses first model found.
    label_conversion : dic | None, default: None
        A dictionary for converting the predicted labels (which are integers) to custom labels. If None,
        tries to extract from `model_info.json` file. The dictionary should have the format {old_label: new_label}.
    export_to_phy : bool, default: False
        Whether to export the results to Phy format. Default is False.
    trust_model : bool, default: False
        Whether to trust the model. If True, the `trusted` parameter that is passed to `skops.load` to load the model will be
        automatically inferred. If False, the `trusted` parameter must be provided to indicate the trusted objects.
    trusted : list of str, default: None
        Passed to skops.load. The object will be loaded only if there are only trusted objects and objects of types listed in trusted in the dumped file.
    enforce_metric_params : bool, default: False
            If True and the parameters used to compute the metrics in `sorting_analyzer` are different than the parmeters
            used to compute the metrics used to train the model, this function will raise an error. Otherwise, a warning is raised.


    Returns
    -------
    classified_units : pd.DataFrame
        A dataframe containing the classified units, indexed by the `unit_ids`, containing the predicted label
        and confidence probability of each labelled unit.

    Raises
    ------
    ValueError
        If the pipeline is not an instance of sklearn.pipeline.Pipeline.

    """
    from sklearn.pipeline import Pipeline

    model, model_info = load_model(
        model_folder=model_folder, repo_id=repo_id, model_name=model_name, trust_model=trust_model, trusted=trusted
    )

    if not isinstance(model, Pipeline):
        raise ValueError("The model must be an instance of sklearn.pipeline.Pipeline")

    model_based_classification = ModelBasedClassification(sorting_analyzer, model)

    classified_units = model_based_classification.predict_labels(
        label_conversion=label_conversion,
        export_to_phy=export_to_phy,
        model_info=model_info,
        enforce_metric_params=enforce_metric_params,
    )

    return classified_units


def load_model(model_folder=None, repo_id=None, model_name=None, trust_model=False, trusted=None):
    """
    Loads a model and model_info from a HuggingFaceHub repo or a local folder.

    Parameters
    ----------
    model_folder : str or Path, defualt: None
        The path to the folder containing the model
    repo_id : str | Path, default: None
        Hugging face repo id which contains the model e.g. 'username/model'
    model_name: str | Path, default: None
        Filename of model e.g. 'my_model.skops'. If None, uses first model found.
    trust_model : bool, default: False
        Whether to trust the model. If True, the `trusted` parameter that is passed to `skops.load` to load the model will be
        automatically inferred. If False, the `trusted` parameter must be provided to indicate the trusted objects.
    trusted : list of str, default: None
        Passed to skops.load. The object will be loaded only if there are only trusted objects and objects of types listed in trusted in the dumped file.


    Returns
    -------
    model, model_info
        A model and metadata about the model
    """

    if model_folder is None and repo_id is None:
        raise ValueError("Please provide a 'model_folder' or a 'repo_id'.")
    elif model_folder is not None and repo_id is not None:
        raise ValueError("Please only provide one of 'model_folder' or 'repo_id'.")
    elif model_folder is not None:
        model, model_info = _load_model_from_folder(
            model_folder=model_folder, model_name=model_name, trust_model=trust_model, trusted=trusted
        )
    else:
        model, model_info = _load_model_from_huggingface(
            repo_id=repo_id, model_name=model_name, trust_model=trust_model, trusted=trusted
        )

    return model, model_info


def _load_model_from_huggingface(repo_id=None, model_name=None, trust_model=False, trusted=None):
    """
    Loads a model from a huggingface repo

    Returns
    -------
    model, model_info
        A model and metadata about the model
    """

    from huggingface_hub import list_repo_files
    from huggingface_hub import hf_hub_download

    # get repo filenames
    repo_filenames = list_repo_files(repo_id=repo_id)

    # download all skops and json files to temp directory
    for filename in repo_filenames:
        if Path(filename).suffix in [".skops", ".json"]:
            full_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model_folder = Path(full_path).parent

    model, model_info = _load_model_from_folder(
        model_folder=model_folder, model_name=model_name, trust_model=trust_model, trusted=trusted
    )

    return model, model_info


def _load_model_from_folder(model_folder=None, model_name=None, trust_model=False, trusted=None):
    """
    Loads a model and model_info from a folder

    Returns
    -------
    model, model_info
        A model and metadata about the model
    """

    import skops.io as skio
    from skops.io.exceptions import UntrustedTypesFoundException

    folder = Path(model_folder)
    assert folder.is_dir(), f"The folder {folder}, does not exist."

    # look for any .skops files
    skops_files = list(folder.glob("*.skops"))
    assert len(skops_files) > 0, f"There are no '.skops' files in the folder {folder}"

    if len(skops_files) > 1:
        if model_name is None:
            model_names = [f.name for f in skops_files]
            raise ValueError(
                f"There are more than 1 '.skops' file in folder {folder}. You have to specify "
                f"the file using the 'model_name' argument. Available files:\n{model_names}"
            )
        else:
            skops_file = folder / Path(model_name)
            assert skops_file.is_file(), f"Model file {skops_file} not found."
    elif len(skops_files) == 1:
        skops_file = skops_files[0]

    if trust_model and trusted is None:
        try:
            model = skio.load(skops_file)
        except UntrustedTypesFoundException as e:
            exception_msg = str(e)
            # the exception message contains the list of untrusted objects after a colon and enswith a period
            trusted = eval(exception_msg.split(":")[1][:-1])
    model = skio.load(skops_file, trusted=trusted)

    model_info_path = folder / "model_info.json"
    if not model_info_path.is_file():
        warnings.warn("No 'model_info.json' file found in folder. No metadata can be checked.")
        model_info = None
    else:
        model_info = json.load(open(model_info_path))

    return model, model_info
