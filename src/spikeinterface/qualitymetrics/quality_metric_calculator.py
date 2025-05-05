"""Classes and functions for computing multiple quality metrics."""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
from warnings import warn

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension


from .quality_metric_list import (
    compute_pc_metrics,
    _misc_metric_name_to_func,
    _possible_pc_metric_names,
    qm_compute_name_to_column_names,
    column_name_to_column_dtype,
)
from .misc_metrics import _default_params as misc_metrics_params
from .pca_metrics import _default_params as pca_metrics_params


class ComputeQualityMetrics(AnalyzerExtension):
    """
    Compute quality metrics on a `sorting_analyzer`.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    metric_names : list or None
        List of quality metrics to compute.
    metric_params : dict of dicts or None
        Dictionary with parameters for quality metrics calculation.
        Default parameters can be obtained with: `si.qualitymetrics.get_default_qm_params()`
    skip_pc_metrics : bool, default: False
        If True, PC metrics computation is skipped.
    delete_existing_metrics : bool, default: False
        If True, any quality metrics attached to the `sorting_analyzer` are deleted. If False, any metrics which were previously calculated but are not included in `metric_names` are kept.

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics.

    Notes
    -----
    principal_components are loaded automatically if already computed.
    """

    extension_name = "quality_metrics"
    depend_on = ["templates", "noise_levels"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True

    def _handle_backward_compatibility_on_load(self):
        # For backwards compatibility - this renames qm_params as metric_params
        if (qm_params := self.params.get("qm_params")) is not None:
            self.params["metric_params"] = qm_params
            del self.params["qm_params"]

    def _set_params(
        self,
        metric_names=None,
        metric_params=None,
        qm_params=None,
        peak_sign=None,
        seed=None,
        skip_pc_metrics=False,
        delete_existing_metrics=False,
        metrics_to_compute=None,
    ):
        if qm_params is not None and metric_params is None:
            deprecation_msg = (
                "`qm_params` is deprecated and will be removed in version 0.104.0 Please use metric_params instead"
            )
            metric_params = qm_params
            warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)

        if metric_names is None:
            metric_names = list(_misc_metric_name_to_func.keys())
            # if PC is available, PC metrics are automatically added to the list
            if self.sorting_analyzer.has_extension("principal_components") and not skip_pc_metrics:
                # by default 'nearest_neightbor' is removed because too slow
                pc_metrics = _possible_pc_metric_names.copy()
                pc_metrics.remove("nn_isolation")
                pc_metrics.remove("nn_noise_overlap")
                metric_names += pc_metrics
            # if spike_locations are not available, drift is removed from the list
            if not self.sorting_analyzer.has_extension("spike_locations"):
                if "drift" in metric_names:
                    metric_names.remove("drift")

        metric_params_ = get_default_qm_params()
        for k in metric_params_:
            if metric_params is not None and k in metric_params:
                metric_params_[k].update(metric_params[k])
            if "peak_sign" in metric_params_[k] and peak_sign is not None:
                metric_params_[k]["peak_sign"] = peak_sign

        metrics_to_compute = metric_names
        qm_extension = self.sorting_analyzer.get_extension("quality_metrics")
        if delete_existing_metrics is False and qm_extension is not None:

            existing_metric_names = qm_extension.params["metric_names"]
            existing_metric_names_propagated = [
                metric_name for metric_name in existing_metric_names if metric_name not in metrics_to_compute
            ]
            metric_names = metrics_to_compute + existing_metric_names_propagated

        params = dict(
            metric_names=metric_names,
            peak_sign=peak_sign,
            seed=seed,
            metric_params=metric_params_,
            skip_pc_metrics=skip_pc_metrics,
            delete_existing_metrics=delete_existing_metrics,
            metrics_to_compute=metrics_to_compute,
        )

        return params

    def _select_extension_data(self, unit_ids):
        new_metrics = self.data["metrics"].loc[np.array(unit_ids)]
        new_data = dict(metrics=new_metrics)
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        import pandas as pd

        metric_names = self.params["metric_names"]
        old_metrics = self.data["metrics"]

        all_unit_ids = new_sorting_analyzer.unit_ids
        not_new_ids = all_unit_ids[~np.isin(all_unit_ids, new_unit_ids)]

        # this creates a new metrics dictionary, but the dtype for everything will be
        # object. So we will need to fix this later after computing metrics
        metrics = pd.DataFrame(index=all_unit_ids, columns=old_metrics.columns)
        metrics.loc[not_new_ids, :] = old_metrics.loc[not_new_ids, :]
        metrics.loc[new_unit_ids, :] = self._compute_metrics(
            new_sorting_analyzer, new_unit_ids, verbose, metric_names, **job_kwargs
        )

        # we need to fix the dtypes after we compute everything because we have nans
        # we can iterate through the columns and convert them back to the dtype
        # of the original quality dataframe.
        for column in old_metrics.columns:
            metrics[column] = metrics[column].astype(old_metrics[column].dtype)

        new_data = dict(metrics=metrics)
        return new_data

    def _compute_metrics(self, sorting_analyzer, unit_ids=None, verbose=False, metric_names=None, **job_kwargs):
        """
        Compute quality metrics.
        """
        import pandas as pd

        metric_params = self.params["metric_params"]
        # sparsity = self.params["sparsity"]
        seed = self.params["seed"]

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        n_jobs = job_kwargs["n_jobs"]
        progress_bar = job_kwargs["progress_bar"]

        if unit_ids is None:
            sorting = sorting_analyzer.sorting
            unit_ids = sorting.unit_ids
            non_empty_unit_ids = sorting.get_non_empty_unit_ids()
            empty_unit_ids = unit_ids[~np.isin(unit_ids, non_empty_unit_ids)]
            if len(empty_unit_ids) > 0:
                warnings.warn(
                    f"Units {empty_unit_ids} are empty. Quality metrics will be set to NaN "
                    f"for these units.\n To remove empty units, use `sorting.remove_empty_units()`."
                )
        else:
            non_empty_unit_ids = unit_ids
            empty_unit_ids = []

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

            params = metric_params[metric_name] if metric_name in metric_params else {}
            res = func(sorting_analyzer, unit_ids=non_empty_unit_ids, **params)
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
        if len(pc_metric_names) > 0 and not self.params["skip_pc_metrics"]:
            if not sorting_analyzer.has_extension("principal_components"):
                raise ValueError(
                    "To compute principal components base metrics, the principal components "
                    "extension must be computed first."
                )
            pc_metrics = compute_pc_metrics(
                sorting_analyzer,
                unit_ids=non_empty_unit_ids,
                metric_names=pc_metric_names,
                # sparsity=sparsity,
                progress_bar=progress_bar,
                n_jobs=n_jobs,
                metric_params=metric_params,
                seed=seed,
            )
            for col, values in pc_metrics.items():
                metrics.loc[non_empty_unit_ids, col] = pd.Series(values)

        # add NaN for empty units
        if len(empty_unit_ids) > 0:
            metrics.loc[empty_unit_ids] = np.nan
            # num_spikes is an int and should be 0
            if "num_spikes" in metrics.columns:
                metrics.loc[empty_unit_ids, ["num_spikes"]] = 0

        # we use the convert_dtypes to convert the columns to the most appropriate dtype and avoid object columns
        # (in case of NaN values)
        metrics = metrics.convert_dtypes()

        # we do this because the convert_dtypes infers the wrong types sometimes.
        # the actual types for columns can be found in column_name_to_column_dtype dictionary.
        for column in metrics.columns:
            if column in column_name_to_column_dtype:
                metrics[column] = metrics[column].astype(column_name_to_column_dtype[column])

        return metrics

    def _run(self, verbose=False, **job_kwargs):

        metrics_to_compute = self.params["metrics_to_compute"]
        delete_existing_metrics = self.params["delete_existing_metrics"]

        computed_metrics = self._compute_metrics(
            sorting_analyzer=self.sorting_analyzer,
            unit_ids=None,
            verbose=verbose,
            metric_names=metrics_to_compute,
            **job_kwargs,
        )

        existing_metrics = []
        # here we get in the loaded via the dict only (to avoid full loading from disk after params reset)
        qm_extension = self.sorting_analyzer.extensions.get("quality_metrics", None)
        if (
            delete_existing_metrics is False
            and qm_extension is not None
            and qm_extension.data.get("metrics") is not None
        ):
            existing_metrics = qm_extension.params["metric_names"]

        # append the metrics which were previously computed
        for metric_name in set(existing_metrics).difference(metrics_to_compute):
            # some metrics names produce data columns with other names. This deals with that.
            for column_name in qm_compute_name_to_column_names[metric_name]:
                computed_metrics[column_name] = qm_extension.data["metrics"][column_name]

        self.data["metrics"] = computed_metrics

    def _get_data(self):
        return self.data["metrics"]


register_result_extension(ComputeQualityMetrics)
compute_quality_metrics = ComputeQualityMetrics.function_factory()


def get_quality_metric_list():
    """
    Return a list of the available quality metrics.
    """

    return deepcopy(list(_misc_metric_name_to_func.keys()))


def get_default_qm_params():
    """
    Return default dictionary of quality metrics parameters.

    Returns
    -------
    dict
        Default qm parameters with metric name as key and parameter dictionary as values.
    """
    default_params = {}
    default_params.update(misc_metrics_params)
    default_params.update(pca_metrics_params)
    return deepcopy(default_params)
