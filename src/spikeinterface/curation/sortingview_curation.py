from __future__ import annotations

import warnings

import json
import warnings
import numpy as np
from pathlib import Path

from .curationsorting import CurationSorting
from .curation_format import (
    convert_from_sortingview_curation_format_v0,
    apply_curation,
    curation_label_to_vectors,
    clean_curation_dict,
)


def get_kachery():
    try:
        import kachery as ka

        return ka
    except ImportError:
        try:
            import kachery_cloud as kcl

            warnings.warn("kachery-cloud is deprecated, use kachery instead", DeprecationWarning, stacklevel=2)
            return kcl
        except ImportError:
            raise ImportError(
                "To apply a SortingView manual curation, you need to have kachery installed:\n"
                ">>> pip install kachery\n(kachery-cloud is also supported, but deprecated)"
            )


def apply_sortingview_curation(
    sorting_or_analyzer, uri_or_json, exclude_labels=None, include_labels=None, skip_merge=False, verbose=None
):
    """
    Apply curation from SortingView manual legacy curation format (before the official "curation_format")

    First, merges (if present) are applied. Then labels are loaded and units
    are optionally filtered based on exclude_labels and include_labels.

    Parameters
    ----------
    sorting_or_analyzer : Sorting | SortingAnalyzer
        The sorting or analyzer to be curated
    uri_or_json : str or Path
        The URI curation link from SortingView or the path to the curation json file
    exclude_labels : list, default: None
        Optional list of labels to exclude (e.g. ["reject", "noise"]).
        Mutually exclusive with include_labels
    include_labels : list, default: None
        Optional list of labels to include (e.g. ["accept"]).
        Mutually exclusive with exclude_labels,  by default None
    skip_merge : bool, default: False
        If True, merges are not applied (only labels)
    verbose : None
        Deprecated


    Returns
    -------
    sorting_or_analyzer_curated : BaseSorting
        The curated sorting or analyzer
    """

    if verbose is not None:
        warnings.warn("versobe in apply_sortingview_curation() is deprecated")

    # download
    if Path(uri_or_json).suffix == ".json" and not str(uri_or_json).startswith("gh://"):
        with open(uri_or_json, "r") as f:
            curation_dict = json.load(f)
    else:
        ka = get_kachery()

        try:
            curation_dict = ka.load_json(uri=uri_or_json)
        except:
            raise Exception(f"Could not retrieve curation from SortingView uri: {uri_or_json}")

    # convert to new format
    if "format_version" not in curation_dict:
        curation_dict = convert_from_sortingview_curation_format_v0(curation_dict)

    unit_ids = sorting_or_analyzer.unit_ids

    # this is a hack because it was not in the old format
    curation_dict["unit_ids"] = list(unit_ids)

    if exclude_labels is not None:
        assert include_labels is None, "Use either `include_labels` or `exclude_labels` to filter units."
        manual_labels = curation_label_to_vectors(curation_dict)
        removed_units = []
        for k in exclude_labels:
            remove_mask = manual_labels[k]
            removed_units.extend(unit_ids[remove_mask])
        removed_units = np.unique(removed_units)
        curation_dict["removed_units"] = removed_units

    if include_labels is not None:
        manual_labels = curation_label_to_vectors(curation_dict)
        removed_units = []
        for k in include_labels:
            remove_mask = ~manual_labels[k]
            removed_units.extend(unit_ids[remove_mask])
        removed_units = np.unique(removed_units)
        curation_dict["removed_units"] = removed_units

    if skip_merge:
        curation_dict["merge_unit_groups"] = []

    # cleaner to ensure validity
    curation_dict = clean_curation_dict(curation_dict)

    # apply
    sorting_curated = apply_curation(sorting_or_analyzer, curation_dict, new_id_strategy="join")

    return sorting_curated


# TODO @alessio you remove this after testing
def apply_sortingview_curation_legacy(
    sorting, uri_or_json, exclude_labels=None, include_labels=None, skip_merge=False, verbose=False
):
    """
    Apply curation from SortingView manual curation.
    First, merges (if present) are applied. Then labels are loaded and units
    are optionally filtered based on exclude_labels and include_labels.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object to be curated
    uri_or_json : str or Path
        The URI curation link from SortingView or the path to the curation json file
    exclude_labels : list, default: None
        Optional list of labels to exclude (e.g. ["reject", "noise"]).
        Mutually exclusive with include_labels
    include_labels : list, default: None
        Optional list of labels to include (e.g. ["accept"]).
        Mutually exclusive with exclude_labels,  by default None
    skip_merge : bool, default: False
        If True, merges are not applied (only labels)
    verbose : bool, default: False
        If True, output is verbose

    Returns
    -------
    sorting_curated : BaseSorting
        The curated sorting
    """
    ka = get_kachery()
    curation_sorting = CurationSorting(sorting, make_graph=False, properties_policy="keep")

    # get sorting view curation
    if Path(uri_or_json).suffix == ".json" and not str(uri_or_json).startswith("gh://"):
        with open(uri_or_json, "r") as f:
            sortingview_curation_dict = json.load(f)
    else:
        try:
            sortingview_curation_dict = ka.load_json(uri=uri_or_json)
        except:
            raise Exception(f"Could not retrieve curation from SortingView uri: {uri_or_json}")

    unit_ids_dtype = sorting.unit_ids.dtype

    # STEP 1: merge groups
    labels_dict = sortingview_curation_dict["labelsByUnit"]
    if "mergeGroups" in sortingview_curation_dict and not skip_merge:
        merge_groups = sortingview_curation_dict["mergeGroups"]
        for merge_group in merge_groups:
            # Store labels of units that are about to be merged
            labels_to_inherit = []
            for unit in merge_group:
                labels_to_inherit.extend(labels_dict.get(str(unit), []))
            labels_to_inherit = list(set(labels_to_inherit))  # Remove duplicates

            if verbose:
                print(f"Merging {merge_group}")
            if unit_ids_dtype.kind in ("U", "S"):
                merge_group = [str(unit) for unit in merge_group]
                # if unit dtype is str, set new id as "{unit1}-{unit2}"
                new_unit_id = "-".join(merge_group)
                curation_sorting.merge(merge_group, new_unit_id=new_unit_id)
            else:
                # in this case, the CurationSorting takes care of finding a new unused int
                curation_sorting.merge(merge_group, new_unit_id=None)
                new_unit_id = curation_sorting.max_used_id  # merged unit id
            labels_dict[str(new_unit_id)] = labels_to_inherit

    # STEP 2: gather and apply sortingview curation labels
    # In sortingview, a unit is not required to have all labels.
    # For example, the first 3 units could be labeled as "accept".
    # In this case, the first 3 values of the property "accept" will be True, the rest False

    # Initialize the properties dictionary
    properties = {
        label: np.zeros(len(curation_sorting.current_sorting.unit_ids), dtype=bool)
        for labels in labels_dict.values()
        for label in labels
    }

    # Populate the properties dictionary
    for unit_index, unit_id in enumerate(curation_sorting.current_sorting.unit_ids):
        unit_id_str = str(unit_id)
        if unit_id_str in labels_dict:
            for label in labels_dict[unit_id_str]:
                properties[label][unit_index] = True

    for prop_name, prop_values in properties.items():
        curation_sorting.current_sorting.set_property(prop_name, prop_values)

    if include_labels is not None or exclude_labels is not None:
        units_to_remove = []
        unit_ids = curation_sorting.current_sorting.unit_ids
        assert include_labels or exclude_labels, "Use either `include_labels` or `exclude_labels` to filter units."
        if include_labels:
            for include_label in include_labels:
                units_to_remove.extend(unit_ids[curation_sorting.current_sorting.get_property(include_label) == False])
        if exclude_labels:
            for exclude_label in exclude_labels:
                units_to_remove.extend(unit_ids[curation_sorting.current_sorting.get_property(exclude_label) == True])
        units_to_remove = np.unique(units_to_remove)
        curation_sorting.remove_units(units_to_remove)
    return curation_sorting.current_sorting
