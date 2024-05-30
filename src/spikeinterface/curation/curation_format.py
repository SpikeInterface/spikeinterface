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
    return True
