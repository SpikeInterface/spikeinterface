"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""
import pandas as pd



def calculate_template_features(waveform_extractor, feature_names=None, **kargs):
    """
    Compute template features like: peak_to_valley/peak_trough_ratio/half_width/repolarization_slope/recovery_slope
    
    """
    unit_ids = waveform_extractor.sorting.unit_ids
    if feature_names is None:
        feature_names = list(_feature_name_to_func.keys())
    
    features = pd.DataFrame(index=unit_ids, columns=feature_names)
    
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id)
        
        for feature_name in feature_names:
            func = _feature_name_to_func[feature_name]
            
            value = func(template)
            
            features.at[unit_id, feature_name] = value

    return features


def get_peak_to_valley(template):
    return 0.

def get_peak_trough_ratio(template):
    return 0.

def get_half_width(template):
    return 0.

def get_repolarization_slope(template):
    return 0.

def get_recovery_slope(template):
    return 0.

_feature_name_to_func = {
    'peak_to_valley': get_peak_to_valley,
    'peak_trough_ratio': get_peak_trough_ratio,
    'half_width': get_half_width,
    'repolarization_slope': get_repolarization_slope,
    'recovery_slope': get_recovery_slope,
}



