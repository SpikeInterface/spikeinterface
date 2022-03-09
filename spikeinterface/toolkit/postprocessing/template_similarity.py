import numpy as np

def compute_template_similarity(waveform_extractor, 
                                waveform_extractor_other=None,
                                method='cosine_similarity'):
    """
    Compute similarity between templates with several methods.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    waveform_extractor_other: WaveformExtractor, optional
        A second waveform extractor object
    method: str
        Method name ('cosine_similarity')

    Returns
    -------
    similarity: np.array
        The similarity matrix
    """
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



def check_equal_template_with_distribution_overlap(waveforms0, waveforms1,
                template0=None, template1=None,
                num_shift = 2, quantile_limit=0.8, 
                return_shift=False,
                debug=False):
    """
    Given 2 waveforms set check if they come from the same distribution.
    
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
        
        #~ if debug:
        #~ if debug and equal:
            #~ import matplotlib.pyplot as plt
            #~ fig, axs = plt.subplots(nrows=2)
            #~ ax = axs[0]
            #~ count, bins = np.histogram(scalar_product0, bins=100)
            #~ ax.plot(bins[:-1], count, color='g')
            #~ count, bins = np.histogram(scalar_product1, bins=100)
            #~ ax.plot(bins[:-1], count, color='r')
            #~ ax.axvline(l0)
            #~ ax.axvline(l1)
            #~ ax.set_title(f'equal={equal} shift={shift}')
            #~ ax = axs[1]
            #~ ax.plot(template0.T.flatten())
            #~ ax.plot(template1.T.flatten())
            #~ ax.set_title(f'shift {shift}')
            #~ plt.show()
        
        if equal:
            final_shift = shift - num_shift
            break
    
    if return_shift:
            return equal, final_shift
    else:
        return equal

