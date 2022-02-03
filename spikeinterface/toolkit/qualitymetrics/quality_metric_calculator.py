import numpy as np
import pandas as pd
import shutil
from copy import deepcopy

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

from .quality_metric_list import (_metric_name_to_func,
                                  calculate_pc_metrics, _possible_pc_metric_names)




class QualityMetricCalculator(BaseWaveformExtractorExtension):
    """
    Class to compute quality metrics of spike sorting output.
    
    principal_component is loaded automatically if already computed.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object

    Returns
    -------
    qmc: QualityMetricCalculator

    """
    
    extension_name = 'quality_metrics'
    
    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)
        
        if waveform_extractor.is_extension('principal_components'):
            self.principal_component = waveform_extractor.load_extension('principal_components')
        else:
            self.principal_component = None

        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting
        
        self._metrics = None

    def _set_params(self, metric_names=None, peak_sign='neg',
                    max_spikes_for_nn = 2000, n_neighbors = 6, seed=None):

        if metric_names is None:
            # This is too slow
            # metric_names = list(_metric_name_to_func.keys()) + _possible_pc_metric_names
            
            # So by default we take all metrics and 3 metrics PCA based only
            # 'nearest_neighbor' is really slow and not taken by default
            metric_names = list(_metric_name_to_func.keys()) 
            if self.principal_component is not None:
                metric_names += ['isolation_distance', 'l_ratio', 'd_prime']
        
        params = dict(metric_names=[str(name) for name in metric_names],
                      peak_sign=peak_sign,
                      max_spikes_for_nn=int(max_spikes_for_nn),
                      n_neighbors=int(n_neighbors),
                      seed=int(seed) if seed is not None else None)
        
        return params

    def _specific_load_from_folder(self):
        self._metrics = pd.read_csv(self.extension_folder / 'metrics.csv', index_col=0)

    def _reset(self):
        self._metrics = None
        
    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # filter metrics dataframe
        new_metrics = self._metrics.loc[np.array(unit_ids)]
        new_metrics.to_csv(new_waveforms_folder / self.extension_name / 'metrics.csv')
        
    def compute_metrics(self):
        """
        Computes quality metrics

        Parameters
        ----------
        metric_names: list or None
            List of quality metrics to compute. If None, all metrics are computed
        **kwargs: keyword arguments for quality metrics (TODO)
            max_spikes_for_nn: int
                maximum number of spikes to use per cluster in PCA metrics
            n_neighbors: int
                number of nearest neighbors to check membership of in PCA metrics
            seed: int
                seed for pseudorandom number generator used in PCA metrics (e.g. nn_isolation)

        Returns
        -------
        metrics: pd.DataFrame

        """
        metric_names = self._params['metric_names']

        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids)

        # simple metrics not based on PCs
        for name in metric_names:
            if name in _possible_pc_metric_names:
                continue
            func = _metric_name_to_func[name]
            
            # TODO add for params from differents functions
            kwargs = {k: self._params[k] for k in ('peak_sign',)}
            
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
            if self.principal_component is None:
                raise ValueError('waveform_principal_component must be provied')
            kwargs = {k: self._params[k] for k in ('max_spikes_for_nn', 'n_neighbors', 'seed')}
            pc_metrics = calculate_pc_metrics(self.principal_component, metric_names=pc_metric_names, **kwargs)
            for col, values in pc_metrics.items():
                metrics[col] = pd.Series(values)
        
        self._metrics = metrics
        
        # save to folder
        metrics.to_csv(self.extension_folder / 'metrics.csv')
        
    def get_metrics(self):
        assert self._metrics is not None, "Quality metrics are not computed. Use the 'compute_metrics()' function."
        return self._metrics


WaveformExtractor.register_extension(QualityMetricCalculator)


def compute_quality_metrics(waveform_extractor, load_if_exists=False, 
                            metric_names=None, **params):
    """
    Compute quality metrics on waveform extractor.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor to comput metrics on
    metric_names: list or None
        List of quality metrics to compute. 
    params: keyword arguments for quality metrics

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics
    """
    
    folder = waveform_extractor.folder
    ext_folder = folder / QualityMetricCalculator.extension_name
    if load_if_exists and ext_folder.is_dir():
        qmc = QualityMetricCalculator.load_from_folder(folder)
    else:
        qmc = QualityMetricCalculator(waveform_extractor)
        qmc.set_params(metric_names=metric_names, **params)
        qmc.compute_metrics()
    
    metrics = qmc.get_metrics()
    
    return metrics


def get_quality_metric_list():
    return deepcopy(list(_metric_name_to_func.keys()))
