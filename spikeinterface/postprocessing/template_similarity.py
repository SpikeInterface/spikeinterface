import numpy as np
from ..core import WaveformExtractor
from ..core.waveform_extractor import BaseWaveformExtractorExtension


class TemplateSimilarityCalculator(BaseWaveformExtractorExtension):
    """Compute similarity between templates with several methods.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """
    extension_name = 'similarity'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        self.waveform_extractor = waveform_extractor
        self.template_metrics = None

    def _set_params(self, method='cosine_similarity'):

        params = dict(method=method)

        return params

    def _specific_load_from_folder(self):
        self.similarity = np.load(self.extension_folder / 'similarity.npy')

    def _reset(self):
        self.similarity = None

    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # filter metrics dataframe
        unit_indices = self.waveform_extractor.sorting.ids_to_indices(unit_ids)
        new_similarity = self.similarity[unit_indices][:, unit_indices]
        np.save(new_waveforms_folder / self.extension_name / 'similarity.npy',
                new_similarity)
        
    def run(self):
        similarity = _compute_template_similarity(self.waveform_extractor, method=self._params['method'])
        np.save(self.extension_folder / 'similarity.npy', similarity)
        self.similarity = similarity

    def get_data(self):
        """Get the computed similarity."""

        msg = "Template similarity is not computed. Use the 'run()' function."
        assert self.similarity is not None, msg
        return self.similarity


WaveformExtractor.register_extension(TemplateSimilarityCalculator)


def _compute_template_similarity(waveform_extractor, 
                                 load_if_exists=False,
                                 method='cosine_similarity',
                                 waveform_extractor_other=None):
    import sklearn.metrics.pairwise

    templates = waveform_extractor.get_all_templates()
    s = templates.shape
    if method == 'cosine_similarity':
        templates_flat = templates.reshape(s[0], -1)
        if waveform_extractor_other is not None:
            templates_other = waveform_extractor_other.get_all_templates()
            s_other = templates_other.shape
            templates_other_flat = templates_other.reshape(s_other[0], -1)
            assert len(templates_flat[0]) == len(templates_other_flat[0]), ("Templates from second WaveformExtractor "
                                                                            "don't have the correct shape!")
        else:
            templates_other_flat = None
        similarity = sklearn.metrics.pairwise.cosine_similarity(templates_flat, templates_other_flat)
    # elif method == '':
    else:
        raise ValueError(f'compute_template_similarity(method {method}) not exists')

    return similarity


def compute_template_similarity(waveform_extractor, 
                                load_if_exists=False,
                                method='cosine_similarity',
                                waveform_extractor_other=None):
    """Compute similarity between templates with several methods.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    load_if_exists : bool, optional, default: False
        Whether to load precomputed similarity, if is already exists.
    method: str
        Method name ('cosine_similarity')
    waveform_extractor_other: WaveformExtractor, optional
        A second waveform extractor object

    Returns
    -------
    similarity: np.array
        The similarity matrix
    """
    if waveform_extractor_other is None:
        folder = waveform_extractor.folder
        ext_folder = folder / TemplateSimilarityCalculator.extension_name
        if load_if_exists and ext_folder.is_dir():
            tmc = TemplateSimilarityCalculator.load_from_folder(folder)
        else:
            tmc = TemplateSimilarityCalculator(waveform_extractor)
            tmc.set_params(method=method)
            tmc.run()
        similarity = tmc.get_data()
        return similarity
    else:
        return _compute_template_similarity(waveform_extractor, waveform_extractor_other, method)



def check_equal_template_with_distribution_overlap(waveforms0, waveforms1,
                                                   template0=None, template1=None,
                                                   num_shift = 2, quantile_limit=0.8, 
                                                   return_shift=False):
    """
    Given 2 waveforms sets, check if they come from the same distribution.
    
    This is computed with a simple trick:
    It project all waveforms from each cluster on the normed vector going from
    one template to another, if the cluster are well separate enought we should
    have one distribution around 0 and one distribution around .
    If the distribution overlap too much then then come from the same distribution.
    
    Done by samuel Garcia with an idea of Crhistophe Pouzat.
    This is used internally by tridesclous for auto merge step.
    
    Can be also used as a distance metrics between 2 clusters.

    waveforms0 and waveforms1 have to be spasifyed outside this function.
    
    This is done with a combinaison of shift bewteen the 2 cluster to also check
    if cluster are similar with a sample shift.

    Parameters
    ----------
    waveforms0, waveforms1: numpy array
        Shape (num_spikes, num_samples, num_chans)
        num_spikes are not necessarly the same for custer.
    template0 , template1=None or numpy array
        The average of each cluster.
        If None, then computed.
    num_shift: int default 2
        number of shift on each side to perform.
    quantile_limit: float in [0 1]
        The quantile overlap limit.

    Returns
    -------
    equal: bool
        equal or not
    """
    
    assert waveforms0.shape[1] == waveforms1.shape[1]
    assert waveforms0.shape[2] == waveforms1.shape[2]
    
    if template0 is None:
        template0 = np.mean(waveforms0, axis=0)

    if template1 is None:
        template1 = np.mean(waveforms1, axis=0)
    
    template0_ = template0[num_shift:-num_shift, :]
    width = template0_.shape[0]

    wfs0 = waveforms0[:, num_shift:-num_shift, :].copy()

    equal = False
    final_shift = None
    for shift in range(num_shift*2+1):

        template1_ = template1[shift:width+shift, :]
        vector_0_1 = (template1_ - template0_)
        vector_0_1 /= np.sum(vector_0_1**2)

        wfs1 = waveforms1[:, shift:width+shift, :].copy()
        
        scalar_product0 = np.sum((wfs0 - template0_[np.newaxis,:,:]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        scalar_product1 = np.sum((wfs1 - template0_[np.newaxis,:,:]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        
        l0 = np.quantile(scalar_product0, quantile_limit)
        l1 = np.quantile(scalar_product1, 1 - quantile_limit)
        
        equal = l0 >= l1
        
        if equal:
            final_shift = shift - num_shift
            break
    
    if return_shift:
            return equal, final_shift
    else:
        return equal

