from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from spikeinterface.core import BaseSorting, SortingAnalyzer, apply_merges_to_sorting, apply_splits_to_sorting
from spikeinterface.curation.curation_model import CurationModel
from spikeinterface.core.sorting_tools import generate_unit_ids_for_split


def validate_curation_dict(curation_dict: dict):
    """
    Validate that the curation dictionary given as parameter complies with the format

    The function do not return anything. This raise an error if something is wring in the format.

    Parameters
    ----------
    curation_dict : dict

    """
    # this will validate the format of the curation_dict
    CurationModel(**curation_dict)


def curation_label_to_vectors(curation_dict_or_model: dict | CurationModel):
    """
    Transform the curation dict into dict of vectors.
    For label category with exclusive=True : a column is created and values are the unique label.
    For label category with exclusive=False : one column per possible is created and values are boolean.

    If exclusive=False and the same label appear several times then it raises an error.

    Parameters
    ----------
    curation_dict : dict or CurationModel
        A curation dictionary or model

    Returns
    -------
    labels : dict of numpy vector

    """
    if isinstance(curation_dict_or_model, dict):
        curation_model = CurationModel(**curation_dict_or_model)
    else:
        curation_model = curation_dict_or_model
    unit_ids = list(curation_model.unit_ids)
    n = len(unit_ids)

    labels = {}

    for label_key, label_def in curation_model.label_definitions.items():
        if label_def.exclusive:
            assert label_key not in labels, f"{label_key} is already a key"
            labels[label_key] = [""] * n
            for manual_label in curation_model.manual_labels:
                values = manual_label.labels.get(label_key, [])
                if len(values) == 1:
                    unit_index = unit_ids.index(manual_label.unit_id)
                    labels[label_key][unit_index] = values[0]
            labels[label_key] = np.array(labels[label_key])
        else:
            for label_opt in label_def.label_options:
                assert label_opt not in labels, f"{label_opt} is already a key"
                labels[label_opt] = np.zeros(n, dtype=bool)
            for manual_label in curation_model.manual_labels:
                values = manual_label.labels.get(label_key, [])
                for value in values:
                    unit_index = unit_ids.index(manual_label.unit_id)
                    labels[value][unit_index] = True
    return labels


def curation_label_to_dataframe(curation_dict_or_model: dict | CurationModel):
    """
    Transform the curation dict into a pandas dataframe.
    For label category with exclusive=True : a column is created and values are the unique label.
    For label category with exclusive=False : one column per possible is created and values are boolean.

    If exclusive=False and the same label appears several times then an error is raised.

    Parameters
    ----------
    curation_dict : dict
        A curation dictionary

    Returns
    -------
    labels : pd.DataFrame
        dataframe with labels.
    """
    import pandas as pd

    if isinstance(curation_dict_or_model, dict):
        curation_model = CurationModel(**curation_dict_or_model)
    else:
        curation_model = curation_dict_or_model

    labels = pd.DataFrame(curation_label_to_vectors(curation_model), index=curation_model.unit_ids)
    return labels


def apply_curation_labels(
    sorting_or_analyzer: BaseSorting | SortingAnalyzer, curation_dict_or_model: dict | CurationModel
):
    """
    Apply manual labels after merges/splits.

    Rules:
      * label for non merged units is applied first
      * for merged group, when exclusive=True, if all have the same label then this label is applied
      * for merged group, when exclusive=False, if one unit has the label then the new one have also it
      * for split units, the original label is applied to all split units
    """
    if isinstance(curation_dict_or_model, dict):
        curation_model = CurationModel(**curation_dict_or_model)
    else:
        curation_model = curation_dict_or_model

    if isinstance(sorting_or_analyzer, BaseSorting):
        sorting = sorting_or_analyzer
    else:
        sorting = sorting_or_analyzer.sorting

    # Please note that manual_labels is done on the unit_ids before the merge!!!
    manual_labels = curation_label_to_vectors(curation_model)

    for key, values in manual_labels.items():
        all_values = np.zeros(sorting.unit_ids.size, dtype=values.dtype)
        for unit_ind, unit_id in enumerate(sorting.unit_ids):
            # if unit_id not in merged_split_units:
            ind = list(curation_model.unit_ids).index(unit_id)
            all_values[unit_ind] = values[ind]
        sorting.set_property(key, all_values)


def apply_curation(
    sorting_or_analyzer: BaseSorting | SortingAnalyzer,
    curation_dict_or_model: dict | CurationModel,
    censor_ms: float | None = None,
    new_id_strategy: str = "append",
    merging_mode: str = "soft",
    sparsity_overlap: float = 0.75,
    raise_error_if_overlap_fails: bool = True,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Apply curation dict to a Sorting or a SortingAnalyzer.

    Steps are done in this order:

      1. Apply labels using curation_dict["manual_labels"]
      2. Remove whole units using curation_dict["removed"]
      3. Apply splits using curation_dict["splits"] and remove spikes from units using curation_dict["discard_spikes"]
      4. Apply merges using curation_dict["merges"]

    A new Sorting or SortingAnalyzer (in memory) is returned.
    The user (an adult) has the responsability to save it somewhere (or not).

    Parameters
    ----------
    sorting_or_analyzer : Sorting | SortingAnalyzer
        The Sorting or SortingAnalyzer object to apply merges.
    curation_dict : dict or CurationModel
        The curation dict or model.
    censor_ms : float | None, default: None
        When applying the merges, any consecutive spikes within the `censor_ms` are removed. This can be thought of
        as the desired refractory period. If `censor_ms=None`, no spikes are discarded.
    new_id_strategy : "append" | "take_first" | "join", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorting.unit_ids)
            * "take_first" : new_unit_ids will be the first unit_id of every list of merges
            * "join" : new_unit_ids will be the concatenation of all unit_ids of every list of merges
    merging_mode : "soft" | "hard", default: "soft"
        How merges are performed for SortingAnalyzer. If the `merge_mode` is "soft" , merges will be approximated, with no reloading of
        the waveforms. This will lead to approximations. If `merge_mode` is "hard", recomputations are accurately
        performed, reloading waveforms if needed
    sparsity_overlap : float, default 0.75
        The percentage of overlap that units should share in order to accept merges. If this criteria is not
        achieved, soft merging will not be possible and an error will be raised. This is for use with a SortingAnalyzer input.
    raise_error_if_overlap_fails : bool, default: True
        If True and `sparsity_overlap` fails for any unit merges, this will raise an error. If False, units which fail the
        `sparsity_overlap` threshold will be skipped in the merge.
    verbose : bool, default: False
        If True, output is verbose
    **job_kwargs : dict
        Job keyword arguments for `merge_units`

    Returns
    -------
    curated_sorting_or_analyzer : Sorting | SortingAnalyzer
        The curated object.
    """
    assert isinstance(
        sorting_or_analyzer, (BaseSorting, SortingAnalyzer)
    ), f"`sorting_or_analyzer` must be a Sorting or a SortingAnalyzer, not an object of type {type(sorting_or_analyzer)}"
    assert isinstance(
        curation_dict_or_model, (dict, CurationModel)
    ), f"`curation_dict_or_model` must be a dict or a CurationModel, not an object of type {type(curation_dict_or_model)}"
    if isinstance(curation_dict_or_model, dict):
        curation_model = CurationModel(**curation_dict_or_model)
    else:
        curation_model = curation_dict_or_model.model_copy(deep=True)

    if not np.array_equal(np.asarray(curation_model.unit_ids), sorting_or_analyzer.unit_ids):
        raise ValueError("unit_ids from the curation_dict do not match the one from Sorting or SortingAnalyzer")

    # 1. Apply labels
    apply_curation_labels(sorting_or_analyzer, curation_model)

    # 2. Remove units
    if len(curation_model.removed) > 0:
        curated_sorting_or_analyzer = sorting_or_analyzer.remove_units(curation_model.removed)
    else:
        curated_sorting_or_analyzer = sorting_or_analyzer

    # 3. Split and discard spikes from units
    # Do this at the same time, otherwise have to do a lot of spike index shuffling.
    # Strategy: put the discarded spikes in a new unit when splitting, then remove them at the end.
    if len(curation_model.splits) > 0 or len(curation_model.discard_spikes) > 0:
        if len(curation_model.splits) > 0:
            split_spikes_unit_ids = [split.unit_id for split in curation_model.splits]
        if len(curation_model.discard_spikes) > 0:
            discard_spikes_unit_ids = [discard_spike.unit_id for discard_spike in curation_model.discard_spikes]

        split_units = {}

        sorting = (
            curated_sorting_or_analyzer if isinstance(sorting_or_analyzer, BaseSorting) else sorting_or_analyzer.sorting
        )

        for unit_id in curation_model.unit_ids:

            if unit_id in split_spikes_unit_ids:

                split_spikes_arg = np.where(np.array(split_spikes_unit_ids) == unit_id)[0][0]
                split = curation_model.splits[split_spikes_arg]
                split_units[unit_id] = split.get_full_spike_indices(sorting)

            # If the unit is not split, but does contain spikes to discard, make an initial "split"
            # unit containing the full spike train.
            elif unit_id in discard_spikes_unit_ids and unit_id not in split_spikes_unit_ids:

                split_units[unit_id] = [sorting.get_unit_spike_train(unit_id)]

            # Now find all spikes which are in discard_spikes, and remove them from the units-to-split.
            # Put the discarded spikes in their own split-unit, at the end of the list of split units.
            if unit_id in discard_spikes_unit_ids:

                discard_spikes_arg = np.where(np.array(discard_spikes_unit_ids) == unit_id)[0][0]
                discard_spikes = np.array(curation_model.discard_spikes[discard_spikes_arg].indices)

                split_units_with_discard = []
                for split_spike_train in split_units[split.unit_id]:
                    split_spike_train_cleaned = np.setdiff1d(split_spike_train, discard_spikes, assume_unique=True)
                    split_units_with_discard.append(split_spike_train_cleaned)
                split_units_with_discard.append(discard_spikes)
                split_units[unit_id] = split_units_with_discard

        split_new_unit_ids = [s.new_unit_ids for s in curation_model.splits if s.new_unit_ids is not None]
        unit_ids_to_discard = []

        # We need to know which units to remove, so need control of the new unit ids here
        if len(split_new_unit_ids) == 0:
            split_new_unit_ids = None
            new_unit_ids = generate_unit_ids_for_split(
                sorting.unit_ids, split_units, new_unit_ids=None, new_id_strategy=new_id_strategy
            )
            for old_unit_id, new_unit_id_list in zip(split_units.keys(), new_unit_ids):
                if old_unit_id in discard_spikes_unit_ids:
                    unit_ids_to_discard.append(new_unit_id_list[-1])

        if isinstance(sorting_or_analyzer, BaseSorting):
            curated_sorting_or_analyzer = apply_splits_to_sorting(
                curated_sorting_or_analyzer,
                split_units,
                new_unit_ids=split_new_unit_ids,
            )
        else:
            curated_sorting_or_analyzer = curated_sorting_or_analyzer.split_units(
                split_units,
                new_id_strategy=new_id_strategy,
                new_unit_ids=split_new_unit_ids,
                format="memory",
                verbose=verbose,
            )
        if len(unit_ids_to_discard) > 0:
            curated_sorting_or_analyzer = sorting_or_analyzer.remove_units(unit_ids_to_discard)

    # 4. Merge units
    if len(curation_model.merges) > 0:
        merge_unit_groups = [m.unit_ids for m in curation_model.merges]
        merge_new_unit_ids = [m.new_unit_id for m in curation_model.merges if m.new_unit_id is not None]
        if len(merge_new_unit_ids) == 0:
            merge_new_unit_ids = None
        if isinstance(sorting_or_analyzer, BaseSorting):
            curated_sorting_or_analyzer, _, _ = apply_merges_to_sorting(
                curated_sorting_or_analyzer,
                merge_unit_groups=merge_unit_groups,
                censor_ms=censor_ms,
                new_id_strategy=new_id_strategy,
                return_extra=True,
            )
        else:
            curated_sorting_or_analyzer, _ = curated_sorting_or_analyzer.merge_units(
                merge_unit_groups=merge_unit_groups,
                censor_ms=censor_ms,
                merging_mode=merging_mode,
                sparsity_overlap=sparsity_overlap,
                raise_error_if_overlap_fails=raise_error_if_overlap_fails,
                new_id_strategy=new_id_strategy,
                return_new_unit_ids=True,
                format="memory",
                verbose=verbose,
                **job_kwargs,
            )

    return curated_sorting_or_analyzer


def load_curation(curation_path: str | Path) -> CurationModel:
    """
    Loads a curation from a local json file.

    Parameters
    ----------
    curation_path : str or Path
        The path to the curation json file

    Returns
    -------
    curation_model : CurationModel
        A CurationModel object
    """
    with open(curation_path) as f:
        curation_dict = json.load(f)
    return CurationModel(**curation_dict)
