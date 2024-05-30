from itertools import combinations


def validate_curation_dict(curation_dict):
    """
    Validate that the curation dictionary given as parameter complies with the format

    Parameters
    ----------
    curation_dict : dict


    Returns
    -------

    """

    supported_versions = {1}
    unit_set = set(curation_dict["unit_ids"])
    labeled_unit_set = set([lbl["unit_id"] for lbl in curation_dict["manual_labels"]])
    merged_units_set = set(sum(curation_dict["merged_unit_groups"], []))
    removed_units_set = set(curation_dict["removed_units"])
    if not labeled_unit_set.issubset(unit_set):
        raise ValueError("Some labeled units are not in the unit list")
    if not merged_units_set.issubset(unit_set):
        raise ValueError("Some merged units are not in the unit list")
    if not removed_units_set.issubset(unit_set):
        raise ValueError("Some removed units are not in the unit list")
    all_merging_groups = [set(group) for group in curation_dict["merged_unit_groups"]]
    for gp_1, gp_2 in combinations(all_merging_groups, 2):
        if len(gp_1.intersection(gp_2)) != 0:
            raise ValueError("Some units belong to multiple merge groups")
    if len(removed_units_set.intersection(merged_units_set)) != 0:
        raise ValueError("Some units were merged and deleted")
    if curation_dict["format_version"] not in supported_versions:
        raise ValueError(
            f"Format version ({curation_dict['format_version']}) not supported. " f"Only {supported_versions} are valid"
        )
    # Check the labels exclusivity
    for lbl in curation_dict["manual_labels"]:
        lbl_key = lbl["label_category"]
        is_exclusive = curation_dict["label_definitions"][lbl_key]["auto_exclusive"]
        if is_exclusive and not isinstance(lbl["labels"], str):
            raise ValueError(f"{lbl_key} are mutually exclusive labels. {lbl['labels']} is invalid")
        elif not is_exclusive and not isinstance(lbl["labels"], list):
            raise ValueError(f"{lbl_key} are not mutually exclusive labels. " f"{lbl['labels']} should be a lists")
    return True


def convert_from_sortingview(sortingview_dict, destination_format=1):
    """
    Converts the sortingview curation format into a curation dictionary
    Couple of caveats:
        * The list of units is not available in the original sortingview dictionary. We set it to None
        * Labels can not be mutually exclusive.
        * Labels have no category, so we regroup them under the "all_labels" category

    Parameters
    ----------
    sortingview_dict : dict
        Dictionary containing the curation information from sortingview
    destination_format : int
        Version of the format to use.
        Default to 1

    Returns
    -------
    curation_dict: dict
        A curation dictionary
    """
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
    for unit_id, l_labels in sortingview_dict["labelsByUnit"].items():
        all_labels.extend(l_labels)
        u_id = unit_id_type(unit_id)
        all_units.append(u_id)
        manual_labels.append({"unit_id": u_id, "label_category": general_cat, "labels": l_labels})
    labels_def = {"all_labels": {"name": "all_labels", "label_options": all_labels, "auto_exclusive": False}}

    curation_dict = {
        "unit_ids": None,
        "label_definitions": labels_def,
        "manual_labels": manual_labels,
        "merged_unit_groups": merge_groups,
        "removed_units": [],
        "format_version": destination_format,
    }

    return curation_dict


if __name__ == "__main__":
    import json

    with open("src/spikeinterface/curation/tests/sv-sorting-curation-str.json") as jf:
        sv_curation = json.load(jf)
    cur_d = convert_from_sortingview(sortingview_dict=sv_curation)
