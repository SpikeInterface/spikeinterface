import numpy as np
import pandas as pd

from .quality_metric_list import _metric_name_to_func

class QualityMetricCalculator:
    """
    
    """
    def __init__(self, waveform_extractor):
        self.waveform_extractor = waveform_extractor
        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting
    
    def compute_metrics(self, metric_names=None, **kargs):
        if metric_names is None:
            metric_names = list(_metric_name_to_func.keys())
        
        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids, columns=metric_names)
        
        for name in metric_names:
            print(name)
            func = _metric_name_to_func[name]
            res = func(self.waveform_extractor, **kargs)
            if isinstance(res, dict):
                # res is a dict convert to series
                metrics[name] = pd.Series(res)
            else:
                # res is a namedtupple with several dict
                # so several columns
                for i, col in enumerate(res._fields):
                    print(col)
                    print(res[i])
                    metrics[col] = pd.Series(res[i])
        
        return metrics
    


def compute_metrics(waveform_extractor, metric_names=None):
    df = QualityMetricCalculator(waveform_extractor).compute_metrics(metric_names)
    return df
