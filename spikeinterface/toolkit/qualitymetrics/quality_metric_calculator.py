from copy import deepcopy
import pandas as pd

from .quality_metric_list import (_metric_name_to_func,
                                  calculate_pc_metrics, _possible_pc_metric_names)


class QualityMetricCalculator:
    """
    Class to compute quality metrics of spike sorting output.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    waveform_principal_component: WaveformPrincipalComponent
        The principal component object (optional - only required if PC-based metrics need to be computed)

    Returns
    -------
    qmc: QualityMetricCalculator

    """

    def __init__(self, waveform_extractor, waveform_principal_component=None):
        self.waveform_extractor = waveform_extractor
        self.waveform_principal_component = waveform_principal_component

        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting

    def compute_metrics(self, metric_names=None, **kwargs):
        """
        Computes quality metrics

        Parameters
        ----------
        metric_names: list or None
            List of quality metrics to compute. If None, all metrics are computed
        **kwargs: keyword arguments for quality metrics (TODO)

        Returns
        -------

        """
        if metric_names is None:
            # metric_names = list(_metric_name_to_func.keys()) + _possible_pc_metric_names
            
            # By default we take all metrics and 3 metrics PCA based only
            # 'nearest_neighbor' is really slow and not taken by default
            metric_names = list(_metric_name_to_func.keys()) +  ['isolation_distance', 'l_ratio', 'd_prime']

        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids)

        # simple metrics not based on PCs
        for name in metric_names:
            if name in _possible_pc_metric_names:
                continue
            func = _metric_name_to_func[name]
            res = func(self.waveform_extractor, **kwargs)
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
            pc_metrics = calculate_pc_metrics(self.waveform_principal_component, metric_names=pc_metric_names)
            for col, values in pc_metrics.items():
                metrics[col] = pd.Series(values)

        return metrics


def compute_quality_metrics(waveform_extractor, metric_names=None, waveform_principal_component=None,
                            **kwargs):
    """

    Parameters
    ----------
    waveform_extractor
    metric_names
    waveform_principal_component
    kwargs

    Returns
    -------

    """
    qmc = QualityMetricCalculator(waveform_extractor, waveform_principal_component)
    df = qmc.compute_metrics(metric_names, **kwargs)
    return df


def get_quality_metric_list():
    return deepcopy(list(_metric_name_to_func.keys()))
