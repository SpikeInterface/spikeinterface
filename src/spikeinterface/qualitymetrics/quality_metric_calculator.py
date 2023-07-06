"""Classes and functions for computing multiple quality metrics."""
import warnings
from copy import deepcopy

import numpy as np

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension

from .quality_metric_list import calculate_pc_metrics, _misc_metric_name_to_func, _possible_pc_metric_names
from .misc_metrics import _default_params as misc_metrics_params
from .pca_metrics import _default_params as pca_metrics_params


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

    extension_name = "quality_metrics"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        if waveform_extractor.has_recording():
            self.recording = waveform_extractor.recording
        else:
            self.recording = None
        self.sorting = waveform_extractor.sorting

    def _set_params(
        self, metric_names=None, qm_params=None, peak_sign=None, seed=None, sparsity=None, skip_pc_metrics=False
    ):
        if metric_names is None:
            metric_names = list(_misc_metric_name_to_func.keys())
            # if PC is available, PC metrics are automatically added to the list
            if self.waveform_extractor.is_extension("principal_components"):
                # by default 'nearest_neightbor' is removed because too slow
                pc_metrics = _possible_pc_metric_names.copy()
                pc_metrics.remove("nn_isolation")
                pc_metrics.remove("nn_noise_overlap")
                metric_names += pc_metrics
            # if spike_locations are not available, drift is removed from the list
            if not self.waveform_extractor.is_extension("spike_locations"):
                if "drift" in metric_names:
                    metric_names.remove("drift")

        qm_params_ = get_default_qm_params()
        for k in qm_params_:
            if qm_params is not None and k in qm_params:
                qm_params_[k].update(qm_params[k])
            if "peak_sign" in qm_params_[k] and peak_sign is not None:
                qm_params_[k]["peak_sign"] = peak_sign

        params = dict(
            metric_names=[str(name) for name in metric_names],
            sparsity=sparsity,
            peak_sign=peak_sign,
            seed=seed,
            qm_params=qm_params_,
            skip_pc_metrics=skip_pc_metrics,
        )

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        new_metrics = self._extension_data["metrics"].loc[np.array(unit_ids)]
        return dict(metrics=new_metrics)

    def _run(self, verbose, **job_kwargs):
        """
        Compute quality metrics.
        """
        metric_names = self._params["metric_names"]
        qm_params = self._params["qm_params"]
        sparsity = self._params["sparsity"]
        seed = self._params["seed"]

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        n_jobs = job_kwargs["n_jobs"]
        progress_bar = job_kwargs["progress_bar"]

        unit_ids = self.sorting.unit_ids
        non_empty_unit_ids = self.sorting.get_non_empty_unit_ids()
        empty_unit_ids = unit_ids[~np.isin(unit_ids, non_empty_unit_ids)]
        if len(empty_unit_ids) > 0:
            warnings.warn(
                f"Units {empty_unit_ids} are empty. Quality metrcs will be set to NaN "
                f"for these units.\n To remove empty units, use `sorting.remove_empty_units()`."
            )

        import pandas as pd

        metrics = pd.DataFrame(index=unit_ids)

        # simple metrics not based on PCs
        for metric_name in metric_names:
            # keep PC metrics for later
            if metric_name in _possible_pc_metric_names:
                continue
            if verbose:
                if metric_name not in _possible_pc_metric_names:
                    print(f"Computing {metric_name}")

            func = _misc_metric_name_to_func[metric_name]

            params = qm_params[metric_name] if metric_name in qm_params else {}
            res = func(self.waveform_extractor, unit_ids=non_empty_unit_ids, **params)
            # QM with uninstall dependencies might return None
            if res is not None:
                if isinstance(res, dict):
                    # res is a dict convert to series
                    metrics.loc[non_empty_unit_ids, metric_name] = pd.Series(res)
                else:
                    # res is a namedtuple with several dict
                    # so several columns
                    for i, col in enumerate(res._fields):
                        metrics.loc[non_empty_unit_ids, col] = pd.Series(res[i])

        # metrics based on PCs
        pc_metric_names = [k for k in metric_names if k in _possible_pc_metric_names]
        if len(pc_metric_names) > 0 and not self._params["skip_pc_metrics"]:
            if not self.waveform_extractor.is_extension("principal_components"):
                raise ValueError("waveform_principal_component must be provied")
            pc_extension = self.waveform_extractor.load_extension("principal_components")
            pc_metrics = calculate_pc_metrics(
                pc_extension,
                unit_ids=non_empty_unit_ids,
                metric_names=pc_metric_names,
                sparsity=sparsity,
                progress_bar=progress_bar,
                n_jobs=n_jobs,
                qm_params=qm_params,
                seed=seed,
            )
            for col, values in pc_metrics.items():
                metrics.loc[non_empty_unit_ids, col] = pd.Series(values)

        # add NaN for empty units
        if len(empty_unit_ids) > 0:
            metrics.loc[empty_unit_ids] = np.nan

        self._extension_data["metrics"] = metrics

    def get_data(self):
        """
        Get the computed metrics.

        Returns
        -------
        metrics : pd.DataFrame
            Dataframe with quality metrics
        """
        msg = "Quality metrics are not computed. Use the 'run()' function."
        assert self._extension_data["metrics"] is not None, msg
        return self._extension_data["metrics"]

    @staticmethod
    def get_extension_function():
        return compute_quality_metrics


WaveformExtractor.register_extension(QualityMetricCalculator)


def compute_quality_metrics(
    waveform_extractor,
    load_if_exists=False,
    metric_names=None,
    qm_params=None,
    peak_sign=None,
    seed=None,
    sparsity=None,
    skip_pc_metrics=False,
    verbose=False,
    **job_kwargs,
):
    """Compute quality metrics on waveform extractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor to compute metrics on.
    load_if_exists : bool, default: False
        Whether to load precomputed quality metrics, if they already exist.
    metric_names : list or None
        List of quality metrics to compute.
    qm_params : dict or None
        Dictionary with parameters for quality metrics calculation.
        Default parameters can be obtained with: `si.qualitymetrics.get_default_qm_params()`
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

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics
    """
    if load_if_exists and waveform_extractor.is_extension(QualityMetricCalculator.extension_name):
        qmc = waveform_extractor.load_extension(QualityMetricCalculator.extension_name)
    else:
        qmc = QualityMetricCalculator(waveform_extractor)
        qmc.set_params(
            metric_names=metric_names,
            qm_params=qm_params,
            peak_sign=peak_sign,
            seed=seed,
            sparsity=sparsity,
            skip_pc_metrics=skip_pc_metrics,
        )
        qmc.run(verbose=verbose, **job_kwargs)

    metrics = qmc.get_data()

    return metrics


def get_quality_metric_list():
    """Get a list of the available quality metrics."""

    return deepcopy(list(_misc_metric_name_to_func.keys()))


def get_default_qm_params():
    """Return default dictionary of quality metrics parameters.

    Returns
    -------
    dict
        Default qm parameters with metric name as key and parameter dictionary as values.
    """
    default_params = {}
    default_params.update(misc_metrics_params)
    default_params.update(pca_metrics_params)
    return deepcopy(default_params)
