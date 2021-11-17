from .basecomparison import BaseComparison
from .basemulticomparison import BaseMultiComparison
from .templatecomparison import TemplateComparison


class MultiTemplateComparison(BaseMultiComparison, BaseComparison):
    """
    Compares multiple waveform extractors using template similarity.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    Parameters
    ----------
    waveform_list: list
        List of waveform extractor objects to be compared
    name_list: list
        List of session names. If not given, sorters are named as 'sess0', 'sess1', 'sess2', etc.
    match_score: float
        Minimum agreement score to match units (default 0.5)
    chance_score: float
        Minimum agreement score to for a possible match (default 0.1)
    verbose: bool
        if True, output is verbose

    Returns
    -------
    multi_template_comparison: MultiTemplateComparison
        MultiTemplateComparison object with the multiple template comparisons
    """

    def __init__(self, waveform_list, name_list=None,
                 match_score=0.8, chance_score=0.3, verbose=False,
                 do_matching=True):
        if name_list is None:
            name_list = [f"sess{i}" for i in range(len(waveform_list))]
        self.waveform_list = waveform_list
        BaseComparison.__init__(self, name_list=name_list, match_score=match_score,
                                chance_score=chance_score, verbose=verbose)
        BaseMultiComparison.__init__(self, waveform_list, name_list=name_list)

        if do_matching:
            self._compute_all()

    def _compare_ij(self, i, j):
        comp = TemplateComparison(self.waveform_list[i], self.waveform_list[j],
                                  we1_name=self.name_list[i],
                                  we2_name=self.name_list[j],
                                  match_score=self.match_score,
                                  verbose=False)
        return comp

    def _populate_nodes(self):
        for i, we in enumerate(self.waveform_list):
            session_name = self.name_list[i]
            for unit_id in we.sorting.get_unit_ids():
                node = session_name, unit_id
                self.graph.add_node(node)


def compare_multiple_templates(*args, **kwargs):
    return MultiTemplateComparison(*args, **kwargs)


compare_multiple_templates.__doc__ = MultiTemplateComparison.__doc__
