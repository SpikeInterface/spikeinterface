import numpy as np
from .curationsorting import CurationSorting


def apply_sortingview_curation(
    sorting,
    uri,
    exclude_labels=None,
    include_labels=None,
    verbose=False
):
    """
    Apply curation from SortingView manual curation.
    First, merges (if present) are applied. Then labels are loaded and units
    are optionally filtered based on exclude_labels and include_labels.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object to be curated
    uri : str
        The URI or JOT curation link from sortingview
    exclude_labels : list, optional
        Optional list of labels to exclude (e.g. ["reject", "noise"]).
        Mutually exclusive with include_labels, by default None
    include_labels : list, optional
        Optional list of labels to include (e.g. ["accept"]).
        Mutually exclusive with exclude_labels,  by default None
    verbose : bool, optional
        If True, output is verbose, by default False

    Returns
    -------
    sorting_curated : BaseSorting
        The curated sorting
    """
    try:
        import kachery_cloud as kcl
    except ImportError:
        raise ImportError(
            "To apply a SortingView manual curation, you need to have sortingview installed: "
            ">>> pip install sortingview"
        )
    curation_sorting = CurationSorting(sorting, make_graph=False, properties_policy="keep")

    # get sorting view curation
    try:
        sorting_curation = kcl.load_json(uri=uri)
    except:
        raise Exception(f"Could not retrieve curation from SortingView uri: {uri}")

    # first, merge groups
    if "mergeGroups" in sorting_curation:
        merge_groups = sorting_curation["mergeGroups"]
        for mg in merge_groups:
            if verbose:
                print(f"Merging {mg}")
            curation_sorting.merge(mg, new_unit_id="-".join(mg))

    # gather and apply properties
    labels_dict = sorting_curation["labelsByUnit"]
    properties = {}
    for _, labels in labels_dict.items():
        for label in labels:
            if label not in properties:
                properties[label] = np.zeros(len(curation_sorting.current_sorting.unit_ids), dtype=bool)

    for u_i, unit_id in enumerate(curation_sorting.current_sorting.unit_ids):
        labels_unit = []
        for unit_label, labels in labels_dict.items():
            if unit_label in unit_id:
                labels_unit.extend(labels)
        for label in labels_unit:
            properties[label][u_i] = True
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
