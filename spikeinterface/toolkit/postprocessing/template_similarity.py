

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
