import numpy as np
import pandas as pd

from .quality_metric_list import (_metric_name_to_func, 
    calculate_pc_metrics, _possible_pc_metric_names)

class QualityMetricCalculator:
    """
    
    """
    def __init__(self, waveform_extractor, waveform_principal_component=None):
        self.waveform_extractor = waveform_extractor
        self.waveform_principal_component = waveform_principal_component
        
        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting
    
    def compute_metrics(self, metric_names=None, **kargs):
        if metric_names is None:
            metric_names = list(_metric_name_to_func.keys()) + _possible_pc_metric_names
        
        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids, columns=metric_names)
        
        # simple metrics not based on PCs
        for name in metric_names:
            if name in _possible_pc_metric_names:
                continue
            func = _metric_name_to_func[name]
            res = func(self.waveform_extractor, **kargs)
            if isinstance(res, dict):
                # res is a dict convert to series
                metrics[name] = pd.Series(res)
            else:
                # res is a namedtupple with several dict
                # so several columns
                for i, col in enumerate(res._fields):
                    metrics[col] = pd.Series(res[i])
        
        # metrics based on PCs
        pc_metric_names = [k for k in metric_names if k in _possible_pc_metric_names]
        if len(pc_metric_names):
            if self.waveform_principal_component is None:
                raise ValueError('waveform_principal_component must be provied')
            pc_metrics = calculate_pc_metrics(self.waveform_principal_component)
            for col, values in pc_metrics.items():
                metrics[col] = pd.Series(values)

        return metrics


def compute_quality_metrics(waveform_extractor, metric_names=None, waveform_principal_component=None):
    df = QualityMetricCalculator(waveform_extractor,
            waveform_principal_component=waveform_principal_component).compute_metrics(metric_names)
    return df
