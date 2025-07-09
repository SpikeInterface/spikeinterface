from __future__ import annotations

import warnings

import json
import warnings
import numpy as np
from pathlib import Path

from .curation_format import (
    apply_curation,
    curation_label_to_vectors,
)
from .curation_model import CurationModel, Merge


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

    if curation_dict is None:
        raise ValueError(f"Could not retrieve curation from SortingView uri: {uri_or_json}")

    unit_ids = sorting_or_analyzer.unit_ids
    curation_dict["unit_ids"] = unit_ids
    curation_model = CurationModel(**curation_dict)

    if skip_merge:
        curation_model.merges = []

    # this is a hack because it was not in the old format
    if exclude_labels is not None:
        assert include_labels is None, "Use either `include_labels` or `exclude_labels` to filter units."
        manual_labels = curation_label_to_vectors(curation_dict)
        removed_units = []
        for k in exclude_labels:
            remove_mask = manual_labels[k]
            removed_units.extend(unit_ids[remove_mask])
        removed_units = np.unique(removed_units)
        curation_model.removed = removed_units

    if include_labels is not None:
        manual_labels = curation_label_to_vectors(curation_dict)
        removed_units = []
        for k in include_labels:
            remove_mask = ~manual_labels[k]
            removed_units.extend(unit_ids[remove_mask])
        removed_units = np.unique(removed_units)
        curation_model.removed = removed_units

    # make merges and removed units
    if len(curation_model.removed) > 0:
        clean_merges = []
        for merge in curation_model.merges:
            clean_merge = []
            for unit_id in merge.unit_ids:
                if unit_id not in curation_model.removed:
                    clean_merge.append(unit_id)
            if len(clean_merge) > 1:
                clean_merges.append(Merge(unit_ids=clean_merge))
        curation_model.merges = clean_merges

    # apply curation
    sorting_curated = apply_curation(sorting_or_analyzer, curation_model, new_id_strategy="join")

    return sorting_curated
