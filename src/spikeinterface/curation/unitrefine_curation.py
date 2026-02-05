import warnings
from pathlib import Path

from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation.model_based_curation import model_based_label_units


def unitrefine_label_units(
    sorting_analyzer: SortingAnalyzer,
    noise_neural_classifier: str | Path | None = None,
    sua_mua_classifier: str | Path | None = None,
):
    """Label units using a cascade of pre-trained classifiers for
    noise/neural unit classification and SUA/MUA classification,
    as shown in the UnitRefine paper (see References).
    The noise/neural classifier is applied first to remove noise units,
    then the SUA/MUA classifier is applied to the remaining units.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting results.
    noise_neural_classifier : str or Path or None, default: None
        The path to the folder containing the model, a full path to a model (".skops")
        or a string to a repo on HuggingFace.
        If None, the noise/neural classification step is skipped.
        Make sure to provide at least one of the two classifiers.
    sua_mua_classifier : str or Path or None, default: None
        The path to the folder containing the model, a full path to a model (".skops")
        or a string to a repo on HuggingFace.
        If None, the SUA/MUA classification step is skipped.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit ids as index and "label"/"probability" as column.

    References
    ----------
    The approach is described in [Jain]_.
    """
    import pandas as pd

    if noise_neural_classifier is None and sua_mua_classifier is None:
        raise ValueError(
            "At least one of noise_neural_classifier or sua_mua_classifier must be provided. "
            "Pre-trained models can be found at https://huggingface.co/collections/SpikeInterface/curation-models or "
            "https://huggingface.co/AnoushkaJain3/models. You can also train models on your own data: "
            "see https://github.com/anoushkajain/UnitRefine for more details."
        )

    if noise_neural_classifier is not None:
        # 1. apply the noise/neural classification and remove noise
        noise_neuron_labels = model_based_label_units(
            sorting_analyzer=sorting_analyzer,
            trust_model=True,
            **get_model_based_classification_kwargs(noise_neural_classifier),
        )
        if set(noise_neuron_labels["prediction"]) != {"noise", "neural"}:
            warnings.warn(
                "The noise/neural classifier did not return the expected labels 'noise' and 'neural'. "
                "Please check the model used for classification."
            )
        noise_units = noise_neuron_labels[noise_neuron_labels["prediction"] == "noise"]
        sorting_analyzer_neural = sorting_analyzer.remove_units(noise_units.index)
    else:
        sorting_analyzer_neural = sorting_analyzer
        noise_units = pd.DataFrame(columns=["prediction", "probability"])

    if sua_mua_classifier is not None:
        # 2. apply the sua/mua classification and aggregate results
        if len(sorting_analyzer.unit_ids) > len(noise_units):
            sua_mua_labels = model_based_label_units(
                sorting_analyzer=sorting_analyzer_neural,
                trust_model=True,
                **get_model_based_classification_kwargs(sua_mua_classifier),
            )
            if set(sua_mua_labels["prediction"]) != {"sua", "mua"}:
                warnings.warn(
                    "The sua/mua classifier did not return the expected labels 'sua' and 'mua'. "
                    "Please check the model used for classification."
                )
            all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
        else:
            all_labels = noise_units
    else:
        all_labels = noise_neuron_labels

    # rename prediction column to label
    all_labels = all_labels.rename(columns={"prediction": "label"})
    return all_labels


def get_model_based_classification_kwargs(model: str | Path) -> dict:
    """Get kwargs for model_based_label_units function based on model parameter.

    Parameters
    ----------
    model : str or Path
        The model argument.

    Returns
    -------
    kwargs : dict
        A dictionary with kwargs for model_based_label_units function based on model parameter.
        This could be `model_folder`, `model_folder` + `model_name` or `repo_id`.
    """
    if Path(model).exists():
        if Path(model).is_dir():
            kwargs = {"model_folder": model}
        else:
            kwargs = {"model_folder": Path(model).parent, "model_name": Path(model).name}
    else:
        kwargs = {"repo_id": model}
    return kwargs
