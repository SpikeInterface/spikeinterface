from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from itertools import chain

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeGoodTimeChunks(AnalyzerExtension):
    """Compute good time chunks.

    Parameters
    ----------
    method : "false_positives_and_negatives" | "user_defined" | "combined"
        

    Returns
    -------
    # dict or array depending on output mode
    good_periods_per_unit : numpy.ndarray
        (n_periods, 4) array with columns: segment_id, unit_id, start_time, end_time
    """

    extension_name = "good_periods_per_unit"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    ## todo: add fp fn parameters (flat kwargs)
    def _set_params(self, method: str = "false_positives_and_negatives", user_defined_periods=None):
        if method in ["false_positives_and_negatives", "combined"]:
            if not self.sorting_analyzer.has_extension("amplitude_scalings"):
                raise ValueError(
                    "ComputeGoodTimeChunks with method 'false_positives_and_negatives' requires 'amplitude_scalings' extension."
                )
        elif method == "user_defined":
            assert user_defined_periods is not None, "user_defined_periods must be provided for method 'user_defined'"
        if method == "combined":
            warnings.warn("ComputeGoodTimeChunks was called with method 'combined', yet user_defined_periods are not passed. Falling back to using false positives and negatives only.")
            method = "false_positives_and_negatives"

        params = dict(method=method, user_defined_periods=user_defined_periods)

        return params

    def _select_extension_data(self, unit_ids):
        new_extension_data = self.data
        return new_extension_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        new_extension_data = self.data
        return new_extension_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_extension_data = self.data
        return new_extension_data

    def _run(self, verbose=False):
        method = self.params["method"]
        flat_args = 0 #TODO extract from method_kwargs

        self.data["isi_histograms"] = isi_histograms
        self.data["bins"] = bins

    def _get_data(self):
        return self.data["isi_histograms"], self.data["bins"]


register_result_extension(ComputeISIHistograms)
compute_isi_histograms = ComputeISIHistograms.function_factory()