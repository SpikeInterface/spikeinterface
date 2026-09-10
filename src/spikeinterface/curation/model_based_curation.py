from pathlib import Path
import json
import warnings
import re
from packaging.version import parse
from copy import deepcopy

import numpy as np

from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation.train_manual_curation import (
    _format_metric_dataframe,
)

# Map old metric column names to new metric column names for backwards compatibility
_BACKWARD_COMPATIBILITY_MAP = {
    "peak_to_valley": {"name": "peak_to_trough_duration"},
    "peak_trough_ratio": {"name": "peak_after_to_trough_ratio", "flip_sign": True},
    "half_width": {"name": "trough_half_width"},
}


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
    unit_ids : list | None
        The list of unit IDs to consider for classification. If None, all units in the sorting analyzer are used.

    Methods
    -------
    predict_labels()
        Predicts the labels for the spike sorting data using the trained model.
    """

    def __init__(self, sorting_analyzer: SortingAnalyzer, pipeline=None, unit_ids: list[str | int] | None = None):
        from sklearn.pipeline import Pipeline

        if not isinstance(pipeline, Pipeline):
            raise ValueError("The `pipeline` must be an instance of sklearn.pipeline.Pipeline")

        if sorting_analyzer is None:
            raise ValueError("`sorting_analyzer` must be provided.")
        self.sorting_analyzer = sorting_analyzer
        self.pipeline = pipeline
        self.required_metrics = pipeline.feature_names_in_
        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids
        self.unit_ids = unit_ids

    def predict_labels(
        self,
        label_conversion: dict[int, str] | None = None,
        metrics=None,
        export_to_phy: bool = False,
        phy_folder: Path | None = None,
        model_info: dict | None = None,
        enforce_metric_params: bool = False,
        set_predictions_as_properties: bool = True,
        input_data=None,
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
        export_to_phy : bool, default: False.
            Whether to export the classified units to Phy format. Default is False.
        phy_folder : Path or None, default: None
            The path to the Phy folder where the classified units will be exported. If None,
            the Phy folder will be inferred from the sorting object. If the sorting object does not have a Phy folder,
            the esport will be skipped.
        model_info : dict or None, default: None
            Dictionary of model info containing provenance of the model.
        enforce_metric_params : bool, default: False
            If True and the parameters used to compute the metrics in `sorting_analyzer` are different than the parmeters
            used to compute the metrics used to train the model, this function will raise an error. Otherwise, a warning is raised.
        set_predictions_as_properties : bool, default: True
            Whether to set the predictions as properties in the sorting object.
            If True, the predicted labels and probabilities will be stored in the 'classifier_label' and 'classifier_probability' properties of the sorting object.
        input_data : deprecated, default: None
            Deprecated parameter. Use `metrics` instead.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the classified units and their corresponding predictions and probabilities,
            indexed by their `unit_ids`.
        """
        import pandas as pd

        if input_data is not None:
            warnings.warn(
                "`input_data` is deprecated. Use the `metrics` argument instead.", FutureWarning, stacklevel=2
            )
            if metrics is None:
                metrics = input_data

        # Get metrics DataFrame for classification
        if metrics is None:
            metrics = self.sorting_analyzer.get_metrics_extension_data()
        else:
            if not isinstance(metrics, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")

        # Restrict metrics to the unit_ids of the sorting_analyzer if available
        if len(metrics) > len(self.unit_ids):
            metrics = metrics.loc[self.unit_ids]

        metrics = _handle_backwards_compatibility_in_metrics(metrics, model_info=model_info)
        metrics = check_required_metrics_are_present(self.required_metrics, metrics)

        if model_info is not None and self.sorting_analyzer is not None:
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

        metrics = _format_metric_dataframe(metrics)

        # Apply classifier
        predictions = self.pipeline.predict(metrics)
        probabilities = self.pipeline.predict_proba(metrics)
        probabilities = np.max(probabilities, axis=1)

        if isinstance(label_conversion, dict):
            if set(predictions).issubset(set(label_conversion.keys())) is False:
                raise ValueError("Labels in predictions do not match those in label_conversion")
            predictions = [label_conversion[label] for label in predictions]

        classified_units = pd.DataFrame(
            zip(predictions, probabilities), columns=["prediction", "probability"], index=self.unit_ids
        )

        # Set predictions and probability as sorting properties
        if set_predictions_as_properties:
            self.sorting_analyzer.set_sorting_property("classifier_label", predictions)
            self.sorting_analyzer.set_sorting_property("classifier_probability", probabilities)

        if export_to_phy:
            if phy_folder is None:
                raise ValueError("Phy folder must be provided using the `phy_folder` parameter.")
            classified_units.to_csv(f"{phy_folder}/cluster_prediction.tsv", sep="\t", index_label="cluster_id")

        return classified_units

    def _check_params_for_classification(self, enforce_metric_params=False, model_info=None):
        """
        Check that quality and template metrics parameters match those used to train the model

        Parameters
        ----------
        enforce_metric_params : bool, default: False
            If True and the parameters used to compute the metrics in `sorting_analyzer` are different than the parmeters
            used to compute the metrics used to train the model, this function will raise an error. Otherwise, a warning is raised.
        model_info : dict, default: None
            Dictionary of model info containing provenance of the model.
        """

        extension_names = ["quality_metrics", "template_metrics"]

        metric_extensions = [self.sorting_analyzer.get_extension(extension_name) for extension_name in extension_names]

        for metric_extension, extension_name in zip(metric_extensions, extension_names):

            # remove the 's' at the end of the extension name
            extension_name = extension_name[:-1]
            model_extension_params = model_info["metric_params"].get(extension_name + "_params")

            if metric_extension is not None and model_extension_params is not None:

                metric_params = metric_extension.params["metric_params"]

                inconsistent_metrics = []
                for metric in model_extension_params["metric_names"]:
                    model_metric_params = model_extension_params.get("metric_params")
                    if model_metric_params is None or metric not in model_metric_params:
                        inconsistent_metrics.append(metric)
                    else:
                        if metric not in metric_params:
                            inconsistent_metrics.append(metric)
                        elif metric_params[metric] != model_metric_params[metric]:
                            warning_message = f"{extension_name} params for {metric} do not match those used to train the model. Parameters can be found in the 'model_info.json' file."
                            if enforce_metric_params is True:
                                raise Exception(warning_message)
                            else:
                                warnings.warn(warning_message)

                if len(inconsistent_metrics) > 0:
                    warning_message = f"Parameters used to compute metrics {inconsistent_metrics}, used to train this model, are unknown."
                    if enforce_metric_params is True:
                        raise Exception(warning_message)
                    else:
                        warnings.warn(warning_message)


def model_based_label_units(
    sorting_analyzer: SortingAnalyzer,
    unit_ids=None,
    model_folder=None,
    repo_id=None,
    model_name=None,
    label_conversion=None,
    trust_model=False,
    trusted=None,
    export_to_phy=False,
    enforce_metric_params=False,
    set_predictions_as_properties=True,
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
    unit_ids : list[str | int] | None, default: None
        A list of unit IDs to consider. If None, all units in the sorting_analyzer will be used.
    model_folder : str or Path, default: None
        The path to the folder containing the model
    repo_id : str, default: None
        Hugging face repo id which contains the model e.g. 'username/model'
    model_name: str, default: None
        Filename of model e.g. 'my_model.skops'. If None, uses first model found.
    label_conversion : dict | None, default: None
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
    set_predictions_as_properties : bool, default: True
        Whether to set the predictions as properties in the sorting object.
        If True, the predicted labels and probabilities will be stored in the 'classifier_label' and 'classifier_probability' properties of the sorting object.


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

    model_based_classification = ModelBasedClassification(
        sorting_analyzer=sorting_analyzer, unit_ids=unit_ids, pipeline=model
    )

    classified_units = model_based_classification.predict_labels(
        label_conversion=label_conversion,
        export_to_phy=export_to_phy,
        model_info=model_info,
        enforce_metric_params=enforce_metric_params,
        set_predictions_as_properties=set_predictions_as_properties,
    )

    return classified_units


def get_required_metrics_from_model(
    model_folder=None, repo_id=None, model_name=None, model=None, trust_model=False, trusted=None
):
    """
    Returns the required metrics for a model, either from a model hosted on HuggingFaceHub or one available in a local folder.

    Parameters
    ----------
    model_folder : str or Path, default: None
        The path to the folder containing the model
    repo_id : str, default: None
        Hugging face repo id which contains the model e.g. 'username/model'
    model_name: str, default: None
        Filename of model e.g. 'my_model.skops'. If None, uses first model found.
    model : sklearn.pipeline.Pipeline, default: None
        A trained sklearn pipeline model. If provided, the required metrics will be extracted from this model
    trust_model : bool, default: False
        Whether to trust the model. If True, the `trusted` parameter that is passed to `skops.load` to load the model will be
        automatically inferred. If False, the `trusted` parameter must be provided to indicate the trusted objects.
    trusted : list of str, default: None
        Passed to skops.load. The object will be loaded only if there are only trusted objects and objects of types listed in trusted in the dumped file.

    Returns
    -------
    required_metrics : list of str
        A list of required metrics for the model.
    """
    from sklearn.pipeline import Pipeline

    if model is None:
        model, _ = load_model(
            model_folder=model_folder, repo_id=repo_id, model_name=model_name, trust_model=trust_model, trusted=trusted
        )

    if not isinstance(model, Pipeline):
        raise ValueError("The model must be an instance of sklearn.pipeline.Pipeline")

    return _handle_backwards_compatibility_in_metric_names(list(model.feature_names_in_))


def load_model(model_folder=None, repo_id=None, model_name=None, trust_model=False, trusted=None):
    """
    Loads a model and model_info from a HuggingFaceHub repo or a local folder.

    Parameters
    ----------
    model_folder : str or Path, default: None
        The path to the folder containing the model
    repo_id : str, default: None
        Hugging face repo id which contains the model e.g. 'username/model'
    model_name: str, default: None
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


def _patch_sklearn_imputer_compatibility(model):
    """Fix SimpleImputer attribute rename from _fill_dtype (sklearn<1.5) to _fit_dtype (sklearn>=1.5)."""
    from sklearn.impute import SimpleImputer

    steps_to_check = []
    if hasattr(model, "steps"):
        steps_to_check = [step for _, step in model.steps]
    elif hasattr(model, "estimators_"):
        steps_to_check = list(model.estimators_)

    for step in steps_to_check:
        if hasattr(step, "steps"):
            _patch_sklearn_imputer_compatibility(step)
        elif isinstance(step, SimpleImputer):
            if hasattr(step, "_fill_dtype") and not hasattr(step, "_fit_dtype"):
                step._fit_dtype = step._fill_dtype
            elif hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = step._fit_dtype


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
            # the exception message contains the list of untrusted objects. The following
            #  search assumes it is the only list in the message.
            string_list = re.search(r"\[(.*?)\]", exception_msg).group()
            trusted = [list_item for list_item in string_list.split("'") if len(list_item) > 2]

    model = skio.load(skops_file, trusted=trusted)
    _patch_sklearn_imputer_compatibility(model)

    model_info_path = folder / "model_info.json"
    if not model_info_path.is_file():
        warnings.warn("No 'model_info.json' file found in folder. No metadata can be checked.")
        model_info = None
    else:
        model_info = json.load(open(model_info_path))

    model_info = _handle_backwards_compatibility_metric_params(model_info)

    return model, model_info


def _handle_backwards_compatibility_metric_params(model_info):
    """
    Handles backwards compatibility in metric parameters for models trained with older versions of SpikeInterface.
    In recent versions, some metric parameters have been changed for clarity.
    """
    if (
        model_info.get("metric_params") is not None
        and model_info.get("metric_params").get("quality_metric_params") is not None
    ):
        if (qm_params := model_info["metric_params"]["quality_metric_params"].get("qm_params")) is not None:
            model_info["metric_params"]["quality_metric_params"]["metric_params"] = qm_params
            del model_info["metric_params"]["quality_metric_params"]["qm_params"]

    if (
        model_info.get("metric_params") is not None
        and model_info.get("metric_params").get("template_metric_params") is not None
    ):
        if (tm_params := model_info["metric_params"]["template_metric_params"].get("metrics_kwargs")) is not None:
            metric_params = {}
            for metric_name in model_info["metric_params"]["template_metric_params"].get("metric_names"):
                metric_params[metric_name] = deepcopy(tm_params)
            model_info["metric_params"]["template_metric_params"]["metric_params"] = metric_params
            del model_info["metric_params"]["template_metric_params"]["metrics_kwargs"]

    return model_info


def _handle_backwards_compatibility_in_metrics(calculated_metrics, model_info):
    """
    Handles backwards compatibility in metric names for models trained with older versions of SpikeInterface.
    In recent versions, some metric names have been changed for clarity. In addition, the sign of some metrics
    has been inverted to maintain consistency.

    Parameters
    ----------
    calculated_metrics : pd.DataFrame
        The DataFrame containing the calculated metrics.
    model_info : dict or None
        Dictionary of model info containing provenance of the model.

    Returns
    -------
    pd.DataFrame
        The DataFrame with updated metric names for compatibility.
    """
    if model_info is None:
        return calculated_metrics
    si_version = model_info["requirements"].get("spikeinterface", None)
    if si_version is not None and parse(si_version) < parse("0.103.2"):
        # If the model was trained with SI version < 0.103.2, we need to rename some metrics
        calculated_metrics = calculated_metrics.copy()
        # We need to rename the metrics and flip sign when needed
        for old_name, updated_dict in _BACKWARD_COMPATIBILITY_MAP.items():
            updated_name = updated_dict["name"]
            if updated_name in calculated_metrics.columns:
                calculated_metrics = calculated_metrics.rename(columns={updated_name: old_name})
                if updated_dict.get("flip_sign", False):
                    calculated_metrics[old_name] = -1 * calculated_metrics[old_name]
    return calculated_metrics


def _handle_backwards_compatibility_in_metric_names(model_metric_names):
    """
    Handles backwards compatibility in metric names for models trained with older versions of SpikeInterface.
    In recent versions, some metric names have been changed for clarity.

    Parameters
    ----------
    model_metric_names : list of str
        The list of metric names used in the model.

    Returns
    -------
    list of str
        The list of updated metric names for compatibility.
    """
    updated_metric_names = []
    for metric_name in model_metric_names:
        if metric_name in _BACKWARD_COMPATIBILITY_MAP:
            updated_metric_names.append(_BACKWARD_COMPATIBILITY_MAP[metric_name]["name"])
        else:
            updated_metric_names.append(metric_name)
    return updated_metric_names


def check_required_metrics_are_present(required_metrics, calculated_metrics):
    # Check all the required metrics have been calculated, preserving the order expected by the pipeline
    if set(required_metrics).issubset(set(calculated_metrics.columns)):
        input_data = calculated_metrics[list(required_metrics)]
    else:
        raise ValueError(
            "Input data does not contain all required metrics for classification",
            f"Missing metrics: {set(required_metrics).difference(calculated_metrics.columns)}",
        )

    return input_data
