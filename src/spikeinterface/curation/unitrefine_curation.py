from pathlib import Path

from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation.model_based_curation import model_based_label_units


def unitrefine_label_units(
    sorting_analyzer: SortingAnalyzer,
    noise_neural_classifier: str | Path | None = "SpikeInterface/UnitRefine_noise_neural_classifier_lightweight",
    sua_mua_classifier: str | Path | None = "SpikeInterface/UnitRefine_sua_mua_classifier_lightweight",
):
    """Label units using UnitRefine, which is a cascade of pre-trained classifiers for
    noise/neural unit classification and SUA/MUA classification.
    The noise/neural classifier is applied first to remove noise units,
    then the SUA/MUA classifier is applied to the remaining units.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object containing the spike sorting results.
    noise_neural_classifier : str or Path or None, default: "SpikeInterface/UnitRefine_noise_neural_classifier_lightweight"
        The path to the folder containing the model or a string to a repo on HuggingFace.
        If None, the noise/neural classification step is skipped.
        By default, it uses a pre-trained lightweight model hosted on HuggingFace that does not require principal
        component analysis (PCA) features.
    sua_mua_classifier : str or Path or None, default: "SpikeInterface/UnitRefine_sua_mua_classifier_lightweight"
        The path to the folder containing the model or a string to a repo on HuggingFace.
        If None, the SUA/MUA classification step is skipped.
        By default, it uses a pre-trained lightweight model hosted on HuggingFace that does not require principal
        component analysis (PCA) features.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit ids as index and "label"/"probability" as column.

    References
    ----------
    The approach is described in [Jain]_.
    """
    import pandas as pd
    import pandas as pd

    if noise_neural_classifier is None and sua_mua_classifier is None:
        raise ValueError("At least one of noise_neural_classifier or sua_mua_classifier must be provided.")

    if noise_neural_classifier is not None:
        # 1. apply the noise/neural classification and remove noise
        noise_neuron_labels = model_based_label_units(
            sorting_analyzer=sorting_analyzer,
            repo_id=noise_neural_classifier,
            trust_model=True,
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
                repo_id=sua_mua_classifier,
                trust_model=True,
            )
            all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
        else:
            all_labels = noise_units
    else:
        all_labels = noise_neuron_labels

    # rename prediction column to label
    all_labels = all_labels.rename(columns={"prediction": "label"})
    return all_labels
