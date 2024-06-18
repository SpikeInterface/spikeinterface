from itertools import combinations


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
    merged_units_set = set(sum(curation_dict["merged_unit_groups"], []))
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

    all_merging_groups = [set(group) for group in curation_dict["merged_unit_groups"]]
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
        "merged_unit_groups": merge_groups,
        "removed_units": [],
    }

    return curation_dict


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

    labels = pd.DataFrame(index=curation_dict["unit_ids"])

    for label_key, label_def in curation_dict["label_definitions"].items():
        if label_def["exclusive"]:
            assert label_key not in labels.columns, f"{label_key} is already a column"
            labels[label_key] = pd.Series(dtype=str)
            labels[label_key][:] = ""
            for lbl in curation_dict["manual_labels"]:
                value = lbl.get(label_key, [])
                if len(value) == 1:
                    labels.at[lbl["unit_id"], label_key] = value[0]
        else:
            for label_opt in label_def["label_options"]:
                assert label_opt not in labels.columns, f"{label_opt} is already a column"
                labels[label_opt] = pd.Series(dtype=bool)
                labels[label_opt][:] = False
            for lbl in curation_dict["manual_labels"]:
                values = lbl.get(label_key, [])
                for value in values:
                    labels.at[lbl["unit_id"], value] = True

    return labels
