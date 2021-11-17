import numpy as np
import pandas as pd

from ..toolkit.postprocessing.template_similarity import compute_template_similarity
from .basecomparison import BaseComparison


class TemplateComparison(BaseComparison):
    """Class to match units based on template similarity

    Parameters
    ----------
    we1 : WaveformExtractor
        The first waveform extractor to get templates to compare
    we2 : WaveformExtractor
        The second waveform extractor to get templates to compare
    unit_ids1 : list, optional
        List of units from we1 to compare, by default None
    unit_ids2 : list, optional
        List of units from we2 to compare, by default None
    similarity_method : str, optional
        Method for the similaroty matrix, by default "cosine_similarity"
    sparsity_dict : dict, optional
        Dictionary for sparsity, by default None
    verbose : bool, optional
        If True, output is verbose, by default False

    Returns
    -------
    comparison : TemplateComparison
        The output TemplateComparison object
    """

    def __init__(self, we1, we2, we1_name=None, we2_name=None, 
                 unit_ids1=None, unit_ids2=None,
                 match_score=0.7, chance_score=0.3, 
                 similarity_method="cosine_similarity", sparsity_dict=None,
                 verbose=False):
        if we1_name is None:
            we1_name = "sess1"
        if we2_name is None:
            we2_name = "sess2"
        name_list = [we1_name, we2_name]
        BaseComparison.__init__(self, name_list=name_list,
                                match_score=match_score, chance_score=chance_score,
                                verbose=verbose)

        self.similarity_method = similarity_method
        self.we1 = we1
        self.we2 = we2
        channel_ids1 = we1.recording.get_channel_ids()
        channel_ids2 = we2.recording.get_channel_ids()

        # two options: all channels are shared or partial channels are shared
        if we1.recording.get_num_channels() != we2.recording.get_num_channels():
            raise NotImplementedError
        if np.any([ch1 != ch2 for (ch1, ch2) in zip(channel_ids1, channel_ids2)]):
            # TODO: here we can check location and run it on the union. Might be useful for reconfigurable probes
            raise NotImplementedError

        self.matches = dict()

        if unit_ids1 is None:
            unit_ids1 = we1.sorting.get_unit_ids()

        if unit_ids2 is None:
            unit_ids2 = we2.sorting.get_unit_ids()
        self.unit_ids = [unit_ids1, unit_ids2]

        if sparsity_dict is not None:
            raise NotImplementedError
        else:
            self.sparsity = None

        self._do_agreement()
        self._do_matching()

    def _do_agreement(self):
        if self._verbose:
            print('Agreement scores...')

        agreement_scores = compute_template_similarity(self.we1, self.we2,
                                                       method=self.similarity_method)
        self.agreement_scores = pd.DataFrame(agreement_scores,
                                             index=self.unit_ids[0],
                                             columns=self.unit_ids[1])


def compare_templates(*args, **kwargs):
    return TemplateComparison(*args, **kwargs)


compare_templates.__doc__ = TemplateComparison.__doc__
