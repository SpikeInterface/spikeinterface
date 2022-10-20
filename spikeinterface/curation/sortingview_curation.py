import numpy as np
from ..core import BaseSorting
from ..core.core_tools import define_function_from_class
from .curationsorting import CurationSorting


class SortingViewCurationSorting(BaseSorting):
    """
    Apply curation from SortingView manual curation.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object to be curated
    uri : str
        The URI or JOT curation link from sortingview
    exclude_labels : _type_, optional
        _description_, by default None
    include_labels : _type_, optional
        _description_, by default None
    make_graph : bool, optional
        _description_, by default False
    properties_policy : str, optional
        _description_, by default 'keep'
    verbose : bool, optional

    Raises
    ------
    ImportError
        _description_
    Exception
        _description_

    Returns
    -------
    """
    def __init__(self, sorting, uri, exclude_labels=None, include_labels=None, 
                 make_graph=False, properties_policy='keep', verbose=True):
        try:
            import kachery_cloud as kcl
        except ImportError:
            raise ImportError("To apply a SortingView manual curation, you need to have sortingview installed: "
                              ">>> pip install sortingview")
        self._verbose = verbose
        self._exclude_labels = exclude_labels
        self._include_labels = include_labels
        self._curation_sorting = CurationSorting(sorting, make_graph=make_graph, 
                                                 properties_policy=properties_policy)

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
                self._curation_sorting.merge(mg, new_unit_id="-".join(mg))
        curated_sorting = self._curation_sorting.current_sorting

        # gather and apply properties
        labels_dict = sorting_curation["labelsByUnit"]
        properties = {}
        for _, labels in labels_dict.items():
            for label in labels:
                if label not in properties:
                    properties[label] = np.zeros(len(curated_sorting.unit_ids), dtype=bool)
        for u_i, unit_id in enumerate(curated_sorting.unit_ids):
            labels_unit = []
            for unit_label, labels in labels_dict.items():
                if unit_label in unit_id:
                    labels_unit.extend(labels)
            for label in labels_unit:
                properties[label][u_i] = True
        for prop_name, prop_values in properties.items():
            curated_sorting.set_property(prop_name, prop_values)

        if include_labels is None and exclude_labels is None:
            curated_unit_ids = curated_sorting.unit_ids
            self = curated_sorting
        else:
            units_to_remove = []
            assert include_labels or exclude_labels
            if include_labels:
                for include_label in include_labels:
                    units_to_remove.extend(curated_sorting.unit_ids[curated_sorting.get_property(include_label) == False])
                units_to_remove = np.unique(units_to_remove)
            if exclude_labels:
                for exclude_label in exclude_labels:
                    units_to_remove.extend(curated_sorting.unit_ids[curated_sorting.get_property(exclude_label) == True])
            units_to_remove = np.unique(units_to_remove)
            self._curation_sorting.remove_units(units_to_remove)
            self = self._curation_sorting.current_sorting

        self._kwargs = dict(
            sorting=sorting.to_dict(),
            uri=uri,
            exclude_labels=exclude_labels,
            include_labels=include_labels, 
            make_graph=make_graph,
            properties_policy=properties_policy,
            verbose=False
        )

        
apply_sortingview_curation = define_function_from_class(source_class=SortingViewCurationSorting,
                                                        name="apply_sortingview_curation")