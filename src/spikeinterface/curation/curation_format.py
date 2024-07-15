from itertools import combinations

import numpy as np

from spikeinterface.core import BaseSorting, SortingAnalyzer, apply_merges_to_sorting

supported_curation_format_versions = {"1"}


def validate_curation_dict(curation_dict):
    """
    Validate that the curation dictionary given as parameter complies with the format

    The function do not return anything. This raise an error if something is wring in the format.

    Parameters
    ----------
    curation_dict : dict

    """

    # format
    if "format_version" not in curation_dict:
        raise ValueError("No version_format")

    if curation_dict["format_version"] not in supported_curation_format_versions:
        raise ValueError(
            f"Format version ({curation_dict['format_version']}) not supported. "
            f"Only {supported_curation_format_versions} are valid"
        )

    # unit_ids
    labeled_unit_set = set([lbl["unit_id"] for lbl in curation_dict["manual_labels"]])
    merged_units_set = set(sum(curation_dict["merge_unit_groups"], []))
    removed_units_set = set(curation_dict["removed_units"])

    if curation_dict["unit_ids"] is not None:
        # old format v0 did not contain unit_ids so this can contains None
        unit_set = set(curation_dict["unit_ids"])
        if not labeled_unit_set.issubset(unit_set):
            raise ValueError("Curation format: some labeled units are not in the unit list")
        if not merged_units_set.issubset(unit_set):
            raise ValueError("Curation format: some merged units are not in the unit list")
        if not removed_units_set.issubset(unit_set):
            raise ValueError("Curation format: some removed units are not in the unit list")

    all_merging_groups = [set(group) for group in curation_dict["merge_unit_groups"]]
    for gp_1, gp_2 in combinations(all_merging_groups, 2):
        if len(gp_1.intersection(gp_2)) != 0:
            raise ValueError("Some units belong to multiple merge groups")
    if len(removed_units_set.intersection(merged_units_set)) != 0:
        raise ValueError("Some units were merged and deleted")

    # Check the labels exclusivity
    for lbl in curation_dict["manual_labels"]:
        for label_key in curation_dict["label_definitions"].keys():
            if label_key in lbl:
                unit_id = lbl["unit_id"]
                label_value = lbl[label_key]
                if not isinstance(label_value, list):
                    raise ValueError(f"Curation format: manual_labels {unit_id} is invalid shoudl be a list")

                is_exclusive = curation_dict["label_definitions"][label_key]["exclusive"]

                if is_exclusive and not len(label_value) <= 1:
                    raise ValueError(
                        f"Curation format: manual_labels {unit_id} {label_key} are exclusive labels. {label_value} is invalid"
                    )


def convert_from_sortingview_curation_format_v0(sortingview_dict, destination_format="1"):
    """
    Converts the old sortingview curation format (v0) into a curation dictionary new format (v1)
    Couple of caveats:
        * The list of units is not available in the original sortingview dictionary. We set it to None
        * Labels can not be mutually exclusive.
        * Labels have no category, so we regroup them under the "all_labels" category

    Parameters
    ----------
    sortingview_dict : dict
        Dictionary containing the curation information from sortingview
    destination_format : str
        Version of the format to use.
        Default to "1"

    Returns
    -------
    curation_dict: dict
        A curation dictionary
    """

    assert destination_format == "1"

    merge_groups = sortingview_dict["mergeGroups"]
    merged_units = sum(merge_groups, [])
    if len(merged_units) > 0:
        unit_id_type = int if isinstance(merged_units[0], int) else str
    else:
        unit_id_type = str
    all_units = []
    all_labels = []
    manual_labels = []
    general_cat = "all_labels"
    for unit_id_, l_labels in sortingview_dict["labelsByUnit"].items():
        all_labels.extend(l_labels)
        # recorver the correct type for unit_id
        unit_id = unit_id_type(unit_id_)
        all_units.append(unit_id)
        manual_labels.append({"unit_id": unit_id, general_cat: l_labels})
    labels_def = {"all_labels": {"name": "all_labels", "label_options": list(set(all_labels)), "exclusive": False}}

    curation_dict = {
        "format_version": destination_format,
        "unit_ids": None,
        "label_definitions": labels_def,
        "manual_labels": manual_labels,
        "merge_unit_groups": merge_groups,
        "removed_units": [],
    }

    return curation_dict


def curation_label_to_vectors(curation_dict):
    """
    Transform the curation dict into dict of vectors.
    For label category with exclusive=True : a column is created and values are the unique label.
    For label category with exclusive=False : one column per possible is created and values are boolean.

    If exclusive=False and the same label appear several times then it raises an error.

    Parameters
    ----------
    curation_dict : dict
        A curation dictionary

    Returns
    -------
    labels: dict of numpy vector

    """
    unit_ids = list(curation_dict["unit_ids"])
    n = len(unit_ids)

    labels = {}

    for label_key, label_def in curation_dict["label_definitions"].items():
        if label_def["exclusive"]:
            assert label_key not in labels, f"{label_key} is already a key"
            labels[label_key] = [""] * n
            for lbl in curation_dict["manual_labels"]:
                value = lbl.get(label_key, [])
                if len(value) == 1:
                    unit_index = unit_ids.index(lbl["unit_id"])
                    labels[label_key][unit_index] = value[0]
            labels[label_key] = np.array(labels[label_key])
        else:
            for label_opt in label_def["label_options"]:
                assert label_opt not in labels, f"{label_opt} is already a key"
                labels[label_opt] = np.zeros(n, dtype=bool)
            for lbl in curation_dict["manual_labels"]:
                values = lbl.get(label_key, [])
                for value in values:
                    unit_index = unit_ids.index(lbl["unit_id"])
                    labels[value][unit_index] = True

    return labels


def curation_label_to_dataframe(curation_dict):
    """
    Transform the curation dict into a pandas dataframe.
    For label category with exclusive=True : a column is created and values are the unique label.
    For label category with exclusive=False : one column per possible is created and values are boolean.

    If exclusive=False and the same label appear several times then it raises an error.

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
    labels = pd.DataFrame(curation_label_to_vectors(curation_dict), index=curation_dict["unit_ids"])
    return labels



def apply_curation_labels(sorting, curation_dict):
    labels = curation_label_to_vectors(curation_dict)
    unit_ids = np.asarray(curation_dict["unit_ids"])
    mask = np.isin(unit_ids, sorting.unit_ids)
    for key, values in labels.items():
        sorting.set_property(key, values[mask], unit_ids[mask])


def apply_curation(sorting_or_analyzer, curation_dict, censor_ms=None, new_id_strategy="append",
                   merging_mode="soft", sparsity_overlap=0.75, verbose=False,
                   **job_kwargs):
    """
    Apply curation dict to a Sorting or an SortingAnalyzer.

    Steps are done this order:
      1. Apply removal using curation_dict["removed_units"]
      2. Apply merges using curation_dict["merge_unit_groups"]
      3. Set labels using curation_dict["manual_labels"]

    A new Sorting or SortingAnalyzer (in memory) is returned.
    The user (an adult) has the responsability to save it somewhere (or not).

    Parameters
    ----------
    sorting_or_analyzer : Sorting | SortingAnalyzer
        The Sorting object to apply merges.
    curation_dict : dict
        The curation dict.
    censor_ms: float | None, default: None
        When applying the merges, should be discard consecutive spikes violating a given refractory per
    new_id_strategy : "append" | "take_first", default: "append"
        The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

            * "append" : new_units_ids will be added at the end of max(sorging.unit_ids)
            * "take_first" : new_unit_ids will be the first unit_id of every list of merges
    merging_mode : "soft" | "hard"
        Used for SortingAnalyzer
    sparsity_overlap:

    verbose: 

    **job_kwargs

    Returns
    -------
    sorting_or_analyzer : Sorting | SortingAnalyzer
        The curated object.


    """
    validate_curation_dict(curation_dict)
    if not np.array_equal(np.asarray(curation_dict["unit_ids"]), sorting_or_analyzer.unit_ids):
        raise ValueError("unit_ids from the curation_dict do not match the one from Sorting or SortingAnalyzer")


    if isinstance(sorting_or_analyzer, BaseSorting):
        sorting = sorting_or_analyzer
        sorting = sorting.remove_units(curation_dict["removed_units"])
        sorting = apply_merges_to_sorting(sorting, curation_dict["merge_unit_groups"],
                                          censor_ms=censor_ms, return_kept=False, new_id_strategy=new_id_strategy)
        apply_curation_labels(sorting, curation_dict)
        return sorting
    
    elif isinstance(sorting_or_analyzer, SortingAnalyzer):
        analyzer = sorting_or_analyzer
        analyzer = analyzer.remove_units(curation_dict["removed_units"])
        analyzer = analyzer.merge_units(
            curation_dict["merge_unit_groups"],
            censor_ms=censor_ms,
            merging_mode=merging_mode,
            sparsity_overlap=sparsity_overlap,
            new_id_strategy=new_id_strategy,
            format="memory",
            verbose=verbose,
            **job_kwargs,
        )
        apply_curation_labels(analyzer.sorting, curation_dict)
        return analyzer
    else:
        raise ValueError("apply_curation() must have a Sorting or a SortingAnalyzer")


