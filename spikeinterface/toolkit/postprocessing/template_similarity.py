

def compute_template_similarity(waveform_extractor, method='cosine_similarity'):
    """
    Compute similarity between templates with several methods.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    method: str
        Method name
    """
    import sklearn.metrics.pairwise

    templates = waveform_extractor.get_all_templates()
    s = templates.shape
    if method == 'cosine_similarity':
        templates_flat = templates.reshape(s[0], -1)
        similarity = sklearn.metrics.pairwise.cosine_similarity(templates_flat)
    # elif method == '':
    else:
        raise ValueError(f'compute_template_similarity(method{method}) not exists')

    return similarity
