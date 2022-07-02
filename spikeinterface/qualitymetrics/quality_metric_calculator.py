"""Classes and functions for computing multiple quality metrics."""

from copy import deepcopy

import numpy as np
import pandas as pd

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

from .quality_metric_list import (_metric_name_to_func,
                                  calculate_pc_metrics, _possible_pc_metric_names)


class QualityMetricCalculator(BaseWaveformExtractorExtension):
    """Class to compute quality metrics of spike sorting output.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object

    Notes
    -----
    principal_components are loaded automatically if already computed.
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

        self.quality_metrics = None

    def _set_params(self, metric_names=None, sparsity=None, peak_sign='neg',
                    max_spikes_for_nn=2000, n_neighbors=6, seed=None,
                    skip_pc_metrics=False):

        if metric_names is None:
            # This is too slow
            # metric_names = list(_metric_name_to_func.keys()) + _possible_pc_metric_names

            # So by default we take all metrics and 3 metrics PCA based only
            # 'nearest_neighbor' is really slow and not taken by default
            metric_names = list(_metric_name_to_func.keys())
            if self.principal_component is not None:
                metric_names += ['isolation_distance', 'l_ratio', 'd_prime']

        params = dict(metric_names=[str(name) for name in metric_names],
                      sparsity=sparsity,
                      peak_sign=peak_sign,
                      max_spikes_for_nn=int(max_spikes_for_nn),
                      n_neighbors=int(n_neighbors),
                      seed=int(seed) if seed is not None else None,
                      skip_pc_metrics=skip_pc_metrics)

        return params

    def _specific_load_from_folder(self):
        self.quality_metrics = pd.read_csv(self.extension_folder / 'metrics.csv', index_col=0)

    def _reset(self):
        self.quality_metrics = None

    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # filter metrics dataframe
        new_metrics = self.quality_metrics.loc[np.array(unit_ids)]
        new_metrics.to_csv(new_waveforms_folder / self.extension_name / 'metrics.csv')

    def run(self, n_jobs, verbose, progress_bar=False):
        """
        Compute quality metrics.
        """

        metric_names = self._params['metric_names']
        sparsity = self._params['sparsity']

        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids)

        # simple metrics not based on PCs
        for name in metric_names:
            if verbose:
                if name not in _possible_pc_metric_names:
                    print(f"Computing {name}")
            if name in _possible_pc_metric_names:
                continue
            func = _metric_name_to_func[name]

            # TODO add for params from different functions
            kwargs = {k: self._params[k] for k in ('peak_sign',)}

            res = func(self.waveform_extractor, **kwargs)
            if isinstance(res, dict):
                # res is a dict convert to series
                metrics[name] = pd.Series(res)
            else:
                # res is a namedtuple with several dict
                # so several columns
                for i, col in enumerate(res._fields):
                    metrics[col] = pd.Series(res[i])

        # metrics based on PCs
        pc_metric_names = [k for k in metric_names if k in _possible_pc_metric_names]
        if len(pc_metric_names) > 0 and not self._params['skip_pc_metrics']:
            if self.principal_component is None:
                raise ValueError('waveform_principal_component must be provied')
            kwargs = {k: self._params[k] for k in ('max_spikes_for_nn', 'n_neighbors', 'seed')}
            pc_metrics = calculate_pc_metrics(self.principal_component,
                                              metric_names=pc_metric_names, 
                                              sparsity=sparsity,
                                              progress_bar=progress_bar, 
                                              n_jobs=n_jobs, **kwargs)
            for col, values in pc_metrics.items():
                metrics[col] = pd.Series(values)

        self.quality_metrics = metrics

        # save to folder
        metrics.to_csv(self.extension_folder / 'metrics.csv')

    def get_data(self):
        """Get the computed metrics."""

        msg = "Quality metrics are not computed. Use the 'run()' function."
        assert self.quality_metrics is not None, msg
        return self.quality_metrics


WaveformExtractor.register_extension(QualityMetricCalculator)


def compute_quality_metrics(waveform_extractor, load_if_exists=False,
                            metric_names=None, sparsity=None, skip_pc_metrics=False, 
                            n_jobs=1, verbose=False, progress_bar=False, **params):
    """Compute quality metrics on waveform extractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor to compute metrics on.
    load_if_exists : bool, optional, default: False
        Whether to load precomputed quality metrics, if they already exist.
    metric_names : list or None
        List of quality metrics to compute.
    sparsity : dict or None
        If given, the sparse channel_ids for each unit in PCA metrics computation. 
        This is used also to identify neighbor units and speed up computations. 
        If None (default) all channels and all units are used for each unit.
    skip_pc_metrics : bool
        If True, PC metrics computation is skipped.
    n_jobs : int
        Number of jobs (used for PCA metrics)
    verbose : bool
        If True, output is verbose.
    progress_bar : bool
        If True, progress bar is shown.
    **params
        Keyword arguments for quality metrics.

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
        qmc.set_params(metric_names=metric_names, sparsity=sparsity, 
                       skip_pc_metrics=skip_pc_metrics, **params)
        qmc.run(n_jobs, verbose, progress_bar=progress_bar)

    metrics = qmc.get_data()

    return metrics


def get_quality_metric_list():
    """Get a list of the available quality metrics."""

    return deepcopy(list(_metric_name_to_func.keys()))
