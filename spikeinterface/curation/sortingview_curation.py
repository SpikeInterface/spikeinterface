from ..core.core_tools import define_function_from_class
from .curationsorting import CurationSorting


class SortingViewCurationSorting(CurationSorting):
    """
    Apply curation from SortingView manual curation.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object to be curated
    uri : str
        The URI or JOT curation link from sortingview

    Returns
    -------
    """
    def __init__(self, sorting, uri, make_graph=False, properties_policy='keep'):
        try:
            import kachery_cloud as kcl
        except ImportError:
            raise ImportError("To apply a SortingView manual curation, you need to have sortingview installed: "
                              ">>> pip install sortingview")
        super().__init__(sorting, make_graph, properties_policy)

        # get sorting view curation
        try:
            sorting_curation = kcl.load_json(uri=uri)
        except:
            raise Exception(f"Could not retrieve curation from SortingView uri: {uri}")


apply_sortingview_curation = define_function_from_class(source_class=SortingViewCurationSorting,
                                                        name="apply_sortingview_curation")