"""
Implement AnalyzerExtension that are essential and imported in core
  * ComputeRandomSpikes
  * ComputeWaveforms
  * ComputeTemplates
Theses two classes replace the WaveformExtractor

It also implements:
  * ComputeNoiseLevels which is very convenient to have
"""

import warnings
import numpy as np
from collections import namedtuple

from .numpyextractors import NumpySorting
from .sortinganalyzer import SortingAnalyzer, AnalyzerExtension, register_result_extension
from .waveform_tools import extract_waveforms_to_single_buffer, estimate_templates_with_accumulator
from .recording_tools import get_noise_levels
from .template import Templates
from .sorting_tools import random_spikes_selection, select_sorting_periods_mask, spike_vector_to_indices
from .job_tools import fix_job_kwargs, split_job_kwargs


class ComputeRandomSpikes(AnalyzerExtension):
    """
    AnalyzerExtension that select somes random spikes.
    This allows for a subsampling of spikes for further calculations and is important
    for managing that amount of memory and speed of computation in the analyzer.

    This will be used by the `waveforms`/`templates` extensions.

    This internally uses `random_spikes_selection()` parameters.

    Parameters
    ----------
    method : "uniform" | "all", default: "uniform"
        The method to select the spikes
    max_spikes_per_unit : int, default: 500
        The maximum number of spikes per unit, ignored if method="all"
    margin_size : int, default: None
        A margin on each border of segments to avoid border spikes, ignored if method="all"
    seed : int or None, default: None
        A seed for the random generator, ignored if method="all"

    Returns
    -------
    random_spike_indices: np.array
        The indices of the selected spikes
    """

    extension_name = "random_spikes"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _run(self, verbose=False):

        self.data["random_spikes_indices"] = random_spikes_selection(
            self.sorting_analyzer.sorting,
            num_samples=self.sorting_analyzer.rec_attributes["num_samples"],
            **self.params,
        )

    def _set_params(self, method="uniform", max_spikes_per_unit=500, margin_size=None, seed=None):
        params = dict(method=method, max_spikes_per_unit=max_spikes_per_unit, margin_size=margin_size, seed=seed)
        return params

    def _select_extension_data(self, unit_ids):
        random_spikes_indices = self.data["random_spikes_indices"]

        spikes = self.sorting_analyzer.sorting.to_spike_vector()

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        selected_mask = np.zeros(spikes.size, dtype=bool)
        selected_mask[random_spikes_indices] = True

        new_data = dict()
        new_data["random_spikes_indices"] = np.flatnonzero(selected_mask[keep_spike_mask])
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        new_data = dict()
        random_spikes_indices = self.data["random_spikes_indices"]
        if keep_mask is None:
            new_data["random_spikes_indices"] = random_spikes_indices.copy()
        else:
            spikes = self.sorting_analyzer.sorting.to_spike_vector()
            selected_mask = np.zeros(spikes.size, dtype=bool)
            selected_mask[random_spikes_indices] = True
            new_data["random_spikes_indices"] = np.flatnonzero(selected_mask[keep_mask])
        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_data = dict()
        new_data["random_spikes_indices"] = self.data["random_spikes_indices"].copy()
        return new_data

    def _get_data(self):
        return self.data["random_spikes_indices"]

    def get_random_spikes(self):
        # utils to get the some_spikes vector
        # use internal cache
        if not hasattr(self, "_some_spikes"):
            spikes = self.sorting_analyzer.sorting.to_spike_vector()
            self._some_spikes = spikes[self.data["random_spikes_indices"]]
        return self._some_spikes

    def get_selected_indices_in_spike_train(self, unit_id, segment_index):
        # useful for WaveformExtractor backwards compatibility
        # In Waveforms extractor "selected_spikes" was a dict (key: unit_id) of list (segment_index) of indices of spikes in spiketrain
        sorting = self.sorting_analyzer.sorting
        random_spikes_indices = self.data["random_spikes_indices"]

        unit_index = sorting.id_to_index(unit_id)
        spikes = sorting.to_spike_vector()
        spike_indices_in_seg = np.flatnonzero(
            (spikes["segment_index"] == segment_index) & (spikes["unit_index"] == unit_index)
        )
        common_element, inds_left, inds_right = np.intersect1d(
            spike_indices_in_seg, random_spikes_indices, return_indices=True
        )
        selected_spikes_in_spike_train = inds_left
        return selected_spikes_in_spike_train


compute_random_spikes = ComputeRandomSpikes.function_factory()
register_result_extension(ComputeRandomSpikes)


class ComputeWaveforms(AnalyzerExtension):
    """
    AnalyzerExtension that extract some waveforms of each units.

    The sparsity is controlled by the SortingAnalyzer sparsity.

    Parameters
    ----------
    ms_before : float, default: 1.0
        The number of ms to extract before the spike events
    ms_after : float, default: 2.0
        The number of ms to extract after the spike events
    dtype : None | dtype, default: None
        The dtype of the waveforms. If None, the dtype of the recording is used.

    Returns
    -------
    waveforms : np.ndarray
        Array with computed waveforms with shape (num_random_spikes, num_samples, num_channels)
    """

    extension_name = "waveforms"
    depend_on = ["random_spikes"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True

    @property
    def nbefore(self):
        return int(self.params["ms_before"] * self.sorting_analyzer.sampling_frequency / 1000.0)

    @property
    def nafter(self):
        return int(self.params["ms_after"] * self.sorting_analyzer.sampling_frequency / 1000.0)

    def _run(self, verbose=False, **job_kwargs):
        self.data.clear()

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        if self.format == "binary_folder":
            # in that case waveforms are extacted directly in files
            file_path = self._get_binary_extension_folder() / "waveforms.npy"
            mode = "memmap"
            copy = False
        else:
            file_path = None
            mode = "shared_memory"
            copy = True

        if self.sparsity is None:
            sparsity_mask = None
        else:
            sparsity_mask = self.sparsity.mask

        all_waveforms = extract_waveforms_to_single_buffer(
            recording,
            some_spikes,
            unit_ids,
            self.nbefore,
            self.nafter,
            mode=mode,
            return_in_uV=self.sorting_analyzer.return_in_uV,
            file_path=file_path,
            dtype=self.params["dtype"],
            sparsity_mask=sparsity_mask,
            copy=copy,
            job_name="compute_waveforms",
            verbose=verbose,
            **job_kwargs,
        )

        self.data["waveforms"] = all_waveforms

    def _set_params(
        self,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        dtype=None,
    ):
        recording = self.sorting_analyzer.recording
        if dtype is None:
            dtype = recording.get_dtype()

        if np.issubdtype(dtype, np.integer) and self.sorting_analyzer.return_in_uV:
            dtype = "float32"

        dtype = np.dtype(dtype)

        params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            dtype=dtype.str,
        )
        return params

    def _select_extension_data(self, unit_ids):
        # random_spikes_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))
        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        # some_spikes = spikes[random_spikes_indices]
        keep_spike_mask = np.isin(some_spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["waveforms"] = self.data["waveforms"][keep_spike_mask, :, :]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        waveforms = self.data["waveforms"]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()
        if keep_mask is not None:
            spike_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()
            valid = keep_mask[spike_indices]
            some_spikes = some_spikes[valid]
            waveforms = waveforms[valid]
        else:
            waveforms = waveforms.copy()

        old_sparsity = self.sorting_analyzer.sparsity
        if old_sparsity is not None:
            # we need a realignement inside each group because we take the channel intersection sparsity
            for group_ids in merge_unit_groups:
                group_indices = self.sorting_analyzer.sorting.ids_to_indices(group_ids)
                group_sparsity_mask = old_sparsity.mask[group_indices, :]
                group_selection = []
                for unit_id in group_ids:
                    unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                    selection = np.flatnonzero(some_spikes["unit_index"] == unit_index)
                    group_selection.append(selection)
                _inplace_sparse_realign_waveforms(waveforms, group_selection, group_sparsity_mask)

            old_num_chans = int(np.max(np.sum(old_sparsity.mask, axis=1)))
            new_num_chans = int(np.max(np.sum(new_sorting_analyzer.sparsity.mask, axis=1)))
            if new_num_chans < old_num_chans:
                waveforms = waveforms[:, :, :new_num_chans]

        return dict(waveforms=waveforms)

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # splitting only affects random spikes, not waveforms
        new_data = dict(waveforms=self.data["waveforms"].copy())
        return new_data

    def get_waveforms_one_unit(self, unit_id, force_dense: bool = False):
        """
        Returns the waveforms of a unit id.

        Parameters
        ----------
        unit_id : int or str
            The unit id to return waveforms for
        force_dense : bool, default: False
            If True, and SortingAnalyzer must be sparse then only waveforms on sparse channels are returned.

        Returns
        -------
        waveforms: np.array
            The waveforms (num_waveforms, num_samples, num_channels).
            In case sparsity is used, only the waveforms on sparse channels are returned.
        """
        sorting = self.sorting_analyzer.sorting
        unit_index = sorting.id_to_index(unit_id)

        waveforms = self.data["waveforms"]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        spike_mask = some_spikes["unit_index"] == unit_index
        wfs = waveforms[spike_mask, :, :]

        if self.sorting_analyzer.sparsity is not None:
            chan_inds = self.sorting_analyzer.sparsity.unit_id_to_channel_indices[unit_id]
            wfs = wfs[:, :, : chan_inds.size]
            if force_dense:
                num_channels = self.sorting_analyzer.get_num_channels()
                dense_wfs = np.zeros((wfs.shape[0], wfs.shape[1], num_channels), dtype=wfs.dtype)
                dense_wfs[:, :, chan_inds] = wfs
                wfs = dense_wfs

        return wfs

    def _get_data(self):
        return self.data["waveforms"]


def _inplace_sparse_realign_waveforms(waveforms, group_selection, group_sparsity_mask):
    # this is used by "waveforms" extension but also "pca"

    # common mask is intersection
    common_mask = np.all(group_sparsity_mask, axis=0)

    for i in range(len(group_selection)):
        chan_mask = group_sparsity_mask[i, :]
        sel = group_selection[i]
        wfs = waveforms[sel, :, :][:, :, : np.sum(chan_mask)]
        keep_mask = common_mask[chan_mask]
        wfs = wfs[:, :, keep_mask]
        waveforms[:, :, : wfs.shape[2]][sel, :, :] = wfs
        waveforms[:, :, wfs.shape[2] :][sel, :, :] = 0.0


compute_waveforms = ComputeWaveforms.function_factory()
register_result_extension(ComputeWaveforms)


class ComputeTemplates(AnalyzerExtension):
    """
    AnalyzerExtension that computes templates (average, std, median, percentile, ...)

    This depends on the "waveforms" extension (`SortingAnalyzer.compute("waveforms")`)

    When the "waveforms" extension is already computed, then the recording is not needed anymore for this extension.

    Note: by default only the average and std are computed. Other operators (std, median, percentile) can be computed on demand
    after the SortingAnalyzer.compute("templates") and then the data dict is updated on demand.

    Parameters
    ----------
    operators: list[str] | list[(str, float)] (for percentile)
        The operators to compute. Can be "average", "std", "median", "percentile"
        If percentile is used, then the second element of the tuple is the percentile to compute.

    Returns
    -------
    templates: np.ndarray
        The computed templates with shape (num_units, num_samples, num_channels)
    """

    extension_name = "templates"
    depend_on = ["random_spikes|waveforms"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True

    def _handle_backward_compatibility_on_load(self):
        if "ms_before" not in self.params:
            # compatibility february 2024 > july 2024
            self.params["ms_before"] = self.params["nbefore"] * 1000.0 / self.sorting_analyzer.sampling_frequency

        if "ms_after" not in self.params:
            # compatibility february 2024 > july 2024
            self.params["ms_after"] = self.params["nafter"] * 1000.0 / self.sorting_analyzer.sampling_frequency

    def _set_params(self, ms_before: float = 1.0, ms_after: float = 2.0, operators=None):
        operators = operators or ["average", "std"]
        assert isinstance(operators, list)
        for operator in operators:
            if isinstance(operator, str):
                if operator not in ("average", "std", "median", "mad"):
                    error_msg = (
                        f"You have entered an operator {operator} in your `operators` argument which is "
                        f"not supported. Please use any of ['average', 'std', 'median', 'mad'] instead."
                    )
                    raise ValueError(error_msg)
            else:
                assert isinstance(operator, (list, tuple))
                assert len(operator) == 2
                assert operator[0] == "percentile"

        waveforms_extension = self.sorting_analyzer.get_extension("waveforms")
        if waveforms_extension is not None:
            ms_before = waveforms_extension.params["ms_before"]
            ms_after = waveforms_extension.params["ms_after"]

        params = dict(
            operators=operators,
            ms_before=ms_before,
            ms_after=ms_after,
        )
        return params

    def _run(self, verbose=False, **job_kwargs):
        self.data.clear()

        if self.sorting_analyzer.has_extension("waveforms"):
            self._compute_and_append_from_waveforms(self.params["operators"])

        else:
            bad_operator_list = [
                operator for operator in self.params["operators"] if operator not in ("average", "std")
            ]
            if len(bad_operator_list) > 0:
                raise ValueError(
                    f"Computing templates with operators {bad_operator_list} requires the 'waveforms' extension"
                )

            recording = self.sorting_analyzer.recording
            sorting = self.sorting_analyzer.sorting
            unit_ids = sorting.unit_ids

            # retrieve spike vector and the sampling
            some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

            return_in_uV = self.sorting_analyzer.return_in_uV

            return_std = "std" in self.params["operators"]
            sparsity_mask = None if self.sparsity is None else self.sparsity.mask
            output = estimate_templates_with_accumulator(
                recording,
                some_spikes,
                unit_ids,
                self.nbefore,
                self.nafter,
                return_in_uV=return_in_uV,
                return_std=return_std,
                sparsity_mask=sparsity_mask,
                verbose=verbose,
                **job_kwargs,
            )

            if return_std:
                templates, stds = output
                data = dict(average=templates, std=stds)
            else:
                templates = output
                data = dict(average=templates)

            if self.sparsity is not None:
                # make average and std dense again
                for k, arr in data.items():
                    dense_arr = self.sparsity.densify_templates(arr)
                    data[k] = dense_arr
            self.data.update(data)

    def _compute_and_append_from_waveforms(self, operators):
        if not self.sorting_analyzer.has_extension("waveforms"):
            raise ValueError(f"Computing templates with operators {operators} requires the 'waveforms' extension")

        unit_ids = self.sorting_analyzer.unit_ids
        channel_ids = self.sorting_analyzer.channel_ids
        waveforms_extension = self.sorting_analyzer.get_extension("waveforms")
        waveforms = waveforms_extension.data["waveforms"]

        num_samples = waveforms.shape[1]

        for operator in operators:
            if isinstance(operator, str) and operator in ("average", "std", "median"):
                key = operator
            elif isinstance(operator, (list, tuple)):
                operator, percentile = operator
                assert operator == "percentile"
                key = f"pencentile_{percentile}"
            else:
                raise ValueError(f"ComputeTemplates: wrong operator {operator}")
            self.data[key] = np.zeros((unit_ids.size, num_samples, channel_ids.size))

        # spikes = self.sorting_analyzer.sorting.to_spike_vector()
        # some_spikes = spikes[self.sorting_analyzer.random_spikes_indices]

        assert self.sorting_analyzer.has_extension(
            "random_spikes"
        ), "compute 'templates' requires the random_spikes extension. You can run sorting_analyzer.compute('random_spikes')"
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()
        for unit_index, unit_id in enumerate(unit_ids):
            spike_mask = some_spikes["unit_index"] == unit_index
            wfs = waveforms[spike_mask, :, :]
            if wfs.shape[0] == 0:
                continue

            for operator in operators:
                if operator == "average":
                    arr = np.average(wfs, axis=0)
                    key = operator
                elif operator == "std":
                    arr = np.std(wfs, axis=0)
                    key = operator
                elif operator == "median":
                    arr = np.median(wfs, axis=0)
                    key = operator
                elif isinstance(operator, (list, tuple)):
                    operator, percentile = operator
                    arr = np.percentile(wfs, percentile, axis=0)
                    key = f"pencentile_{percentile}"

                if self.sparsity is None:
                    self.data[key][unit_index, :, :] = arr
                else:
                    channel_indices = self.sparsity.unit_id_to_channel_indices[unit_id]
                    self.data[key][unit_index, :, :][:, channel_indices] = arr[:, : channel_indices.size]

    @property
    def nbefore(self):
        nbefore = int(self.params["ms_before"] * self.sorting_analyzer.sampling_frequency / 1000.0)
        return nbefore

    @property
    def nafter(self):
        nafter = int(self.params["ms_after"] * self.sorting_analyzer.sampling_frequency / 1000.0)
        return nafter

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        new_data = dict()
        for key, arr in self.data.items():
            new_data[key] = arr[keep_unit_indices, :, :]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):

        all_new_units = new_sorting_analyzer.unit_ids
        new_data = dict()
        counts = self.sorting_analyzer.sorting.count_num_spikes_per_unit()
        for key, arr in self.data.items():
            new_data[key] = np.zeros((len(all_new_units), arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            for unit_index, unit_id in enumerate(all_new_units):
                if unit_id not in new_unit_ids:
                    keep_unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                    new_data[key][unit_index] = arr[keep_unit_index, :, :]
                else:
                    merge_group = merge_unit_groups[list(new_unit_ids).index(unit_id)]
                    keep_unit_indices = self.sorting_analyzer.sorting.ids_to_indices(merge_group)
                    # We do a weighted sum of the templates
                    weights = np.zeros(len(merge_group), dtype=np.float32)
                    for count, merge_unit_id in enumerate(merge_group):
                        weights[count] = counts[merge_unit_id]
                    weights /= weights.sum()
                    new_data[key][unit_index] = (arr[keep_unit_indices, :, :] * weights[:, np.newaxis, np.newaxis]).sum(
                        0
                    )
                    if new_sorting_analyzer.sparsity is not None:
                        chan_ids = new_sorting_analyzer.sparsity.unit_id_to_channel_indices[unit_id]
                        mask = ~np.isin(np.arange(arr.shape[2]), chan_ids)
                        new_data[key][unit_index][:, mask] = 0

        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        if not new_sorting_analyzer.has_extension("waveforms"):
            warnings.warn(
                "Splitting templates without the 'waveforms' extension will simply copy the template of the unit that "
                "was split to the new split units. This is not recommended and may lead to incorrect results. It is "
                "recommended to compute the 'waveforms' extension before splitting, or to use 'hard' splitting mode.",
            )
        new_data = dict()
        for operator, arr in self.data.items():
            # we first copy the unsplit units
            new_array = np.zeros((len(new_sorting_analyzer.unit_ids), arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            new_analyzer_unit_ids = list(new_sorting_analyzer.unit_ids)
            unsplit_unit_ids = [unit_id for unit_id in self.sorting_analyzer.unit_ids if unit_id not in split_units]
            new_indices = np.array([new_analyzer_unit_ids.index(unit_id) for unit_id in unsplit_unit_ids])
            old_indices = self.sorting_analyzer.sorting.ids_to_indices(unsplit_unit_ids)
            new_array[new_indices, ...] = arr[old_indices, ...]

            for split_unit_id, new_splits in zip(split_units, new_unit_ids):
                if new_sorting_analyzer.has_extension("waveforms"):
                    for new_unit_id in new_splits:
                        split_unit_index = new_sorting_analyzer.sorting.id_to_index(new_unit_id)
                        wfs = new_sorting_analyzer.get_extension("waveforms").get_waveforms_one_unit(
                            new_unit_id, force_dense=True
                        )

                        if operator == "average":
                            arr = np.average(wfs, axis=0)
                        elif operator == "std":
                            arr = np.std(wfs, axis=0)
                        elif operator == "median":
                            arr = np.median(wfs, axis=0)
                        elif "percentile" in operator:
                            _, percentile = operator.splot("_")
                            arr = np.percentile(wfs, float(percentile), axis=0)
                        new_array[split_unit_index, ...] = arr
                else:
                    split_unit_index = self.sorting_analyzer.sorting.id_to_index(split_unit_id)
                    old_template = arr[split_unit_index, ...]
                    new_indices = new_sorting_analyzer.sorting.ids_to_indices(new_splits)
                    new_array[new_indices, ...] = np.tile(old_template, (len(new_splits), 1, 1))
            new_data[operator] = new_array
        return new_data

    def _get_data(self, operator="average", percentile=None, outputs="numpy"):
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=... if `operator=percentile`"
            key = f"percentile_{percentile}"

        if key not in self.data.keys():
            error_msg = (
                f"You have entered `operator={key}`, but the only operators calculated are "
                f"{list(self.data.keys())}. Please use one of these as your `operator` in the "
                f"`get_data` function."
            )
            raise ValueError(error_msg)

        templates_array = self.data[key]

        if outputs == "numpy":
            return templates_array
        elif outputs == "Templates":
            return Templates(
                templates_array=templates_array,
                sampling_frequency=self.sorting_analyzer.sampling_frequency,
                nbefore=self.nbefore,
                channel_ids=self.sorting_analyzer.channel_ids,
                unit_ids=self.sorting_analyzer.unit_ids,
                probe=self.sorting_analyzer.get_probe(),
            )
        else:
            raise ValueError("outputs must be `numpy` or `Templates`")

    def get_templates(self, unit_ids=None, operator="average", percentile=None, save=True, outputs="numpy"):
        """
        Return templates (average, std, median or percentiles) for multiple units.

        If not computed yet then this is computed on demand and optionally saved.

        Parameters
        ----------
        unit_ids : list or None
            Unit ids to retrieve waveforms for
        operator : "average" | "median" | "std" | "percentile", default: "average"
            The operator to compute the templates
        percentile : float, default: None
            Percentile to use for operator="percentile"
        save : bool, default: True
            In case, the operator is not computed yet it can be saved to folder or zarr
        outputs : "numpy" | "Templates", default: "numpy"
            Whether to return a numpy array or a Templates object

        Returns
        -------
        templates : np.array | Templates
            The returned templates (num_units, num_samples, num_channels)
        """
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=... if `operator='percentile'`"
            key = f"pencentile_{percentile}"

        if key in self.data:
            templates_array = self.data[key]
        else:
            if operator != "percentile":
                self._compute_and_append_from_waveforms([operator])
                self.params["operators"] += [operator]
            else:
                self._compute_and_append_from_waveforms([(operator, percentile)])
                self.params["operators"] += [(operator, percentile)]
            templates_array = self.data[key]

            if save:
                if not self.sorting_analyzer.is_read_only():
                    self.save()

        if unit_ids is not None:
            unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
            templates_array = templates_array[unit_indices, :, :]
        else:
            unit_ids = self.sorting_analyzer.unit_ids

        if outputs == "numpy":
            return templates_array
        elif outputs == "Templates":
            return Templates(
                templates_array=templates_array,
                sampling_frequency=self.sorting_analyzer.sampling_frequency,
                nbefore=self.nbefore,
                channel_ids=self.sorting_analyzer.channel_ids,
                unit_ids=unit_ids,
                probe=self.sorting_analyzer.get_probe(),
                is_in_uV=self.sorting_analyzer.return_in_uV,
            )
        else:
            raise ValueError("`outputs` must be 'numpy' or 'Templates'")

    def get_unit_template(self, unit_id, operator="average"):
        """
        Return template for a single unit.

        Parameters
        ----------
        unit_id: str | int
            Unit id to retrieve waveforms for
        operator: str, default: "average"
             The operator to compute the templates

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        templates = self.data[operator]
        unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)

        return np.array(templates[unit_index, :, :])


compute_templates = ComputeTemplates.function_factory()
register_result_extension(ComputeTemplates)


class ComputeNoiseLevels(AnalyzerExtension):
    """
    Computes the noise level associated with each recording channel.

    This function will wraps the `get_noise_levels(recording)` to make the noise levels persistent
    on disk (folder or zarr) as a `WaveformExtension`.
    The noise levels do not depend on the unit list, only the recording, but it is a convenient way to
    retrieve the noise levels directly ine the WaveformExtractor.

    Note that the noise levels can be scaled or not, depending on the `return_in_uV` parameter
    of the `SortingAnalyzer`.

    Parameters
    ----------
    **kwargs : dict
        Additional parameters for the `spikeinterface.get_noise_levels()` function

    Returns
    -------
    noise_levels : np.array
        The noise level vector
    """

    extension_name = "noise_levels"
    depend_on = []
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True

    def _set_params(self, **noise_level_params):
        params = noise_level_params.copy()
        return params

    def _select_extension_data(self, unit_ids):
        # this does not depend on units
        return self.data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        # this does not depend on units
        return self.data.copy()

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # this does not depend on units
        return self.data.copy()

    def _run(self, verbose=False, **job_kwargs):
        self.data["noise_levels"] = get_noise_levels(
            self.sorting_analyzer.recording,
            return_in_uV=self.sorting_analyzer.return_in_uV,
            **self.params,
            **job_kwargs,
        )

    def _get_data(self):
        return self.data["noise_levels"]

    def _handle_backward_compatibility_on_load(self):
        # The old parameters used to be params=dict(num_chunks_per_segment=20, chunk_size=10000, seed=None)
        # now it is handle more explicitly using random_slices_kwargs=dict()
        for key in ("num_chunks_per_segment", "chunk_size", "seed"):
            if key in self.params:
                if "random_slices_kwargs" not in self.params:
                    self.params["random_slices_kwargs"] = dict()
                self.params["random_slices_kwargs"][key] = self.params.pop(key)


register_result_extension(ComputeNoiseLevels)
compute_noise_levels = ComputeNoiseLevels.function_factory()


class BaseMetric:
    """
    Base class for metric-based extension
    """

    metric_name = None  # to be defined in subclass
    metric_params = {}  # to be defined in subclass
    metric_columns = {}  # column names and their dtypes of the dataframe
    metric_descriptions = {}  # descriptions of each metric column
    needs_recording = False  # whether the metric needs recording
    needs_tmp_data = False  # whether the metric needs temporary data computed with MetricExtension._prepare_data
    needs_job_kwargs = False  # whether the metric needs job_kwargs
    supports_periods = False  # whether the metric function supports periods
    depend_on = []  # extensions the metric depends on

    # the metric function must have the signature:
    # def metric_function(sorting_analyzer, unit_ids, **metric_params)
    # or if needs_tmp_data=True
    # def metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params)
    # or if needs_job_kwargs=True
    # def metric_function(sorting_analyzer, unit_ids, tmp_data, job_kwargs, **metric_params)
    # and must return a dict ({unit_id: values}) or namedtuple with fields matching metric_columns keys
    metric_function = None  # to be defined in subclass

    @classmethod
    def compute(cls, sorting_analyzer, unit_ids, metric_params, tmp_data, job_kwargs, periods=None):
        """Compute the metric.

        Parameters
        ----------
        sorting_analyzer :  SortingAnalyzer
            The input sorting analyzer
        unit_ids : list
            List of unit ids to compute the metric for
        metric_params : dict
            Parameters to override the default metric parameters
        tmp_data : dict
            Temporary data to pass to the metric function
        job_kwargs : dict
            Job keyword arguments to control parallelization
        periods : np.ndarray | None
            Numpy array of unit periods of unit_period_dtype if supports_periods is True

        Returns
        -------
        results: namedtuple
            The results of the metric function
        """
        args = (sorting_analyzer, unit_ids)
        if cls.needs_tmp_data:
            args += (tmp_data,)
        if cls.needs_job_kwargs:
            args += (job_kwargs,)
        if cls.supports_periods:
            args += (periods,)

        results = cls.metric_function(*args, **metric_params)

        # if namedtuple, check that columns are correct
        if isinstance(results, tuple) and hasattr(results, "_fields"):
            assert set(results._fields) == set(list(cls.metric_columns.keys())), (
                f"Metric {cls.metric_name} returned columns {results._fields} "
                f"but expected columns are {cls.metric_columns.keys()}"
            )
        return results


class BaseMetricExtension(AnalyzerExtension):
    """
    AnalyzerExtension that computes a metric and store the results in a dataframe.

    This depends on one or more extensions (see `depend_on` attribute of the `BaseMetric` subclass).

    Returns
    -------
    metric_dataframe : pd.DataFrame
        The computed metric dataframe.
    """

    extension_name = None  # to be defined in subclass
    metric_class = None  # to be defined in subclass
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = False
    metric_list: list[BaseMetric] = None  # list of BaseMetric

    @classmethod
    def get_available_metric_names(cls):
        """Get the available metric names.

        Returns
        -------
        available_metric_names : list[str]
            List of available metric names.
        """
        return [m.metric_name for m in cls.metric_list]

    @classmethod
    def get_default_metric_params(cls):
        """Get the default metric parameters.

        Returns
        -------
        default_metric_params : dict
            Dictionary of default metric parameters for each metric.
        """
        default_metric_params = {m.metric_name: m.metric_params for m in cls.metric_list}
        return default_metric_params

    @classmethod
    def get_metric_columns(cls, metric_names=None):
        """Get the default metric columns.

        Parameters
        ----------
        metric_names : list[str] | None
            List of metric names to get columns for. If None, all metrics are considered.

        Returns
        -------
        default_metric_columns : dict
            Dictionary of default metric columns and their dtypes for each metric.
        """
        default_metric_columns = []
        for m in cls.metric_list:
            if metric_names is not None and m.metric_name not in metric_names:
                continue
            default_metric_columns.extend(m.metric_columns)
        return default_metric_columns

    @classmethod
    def get_metric_column_descriptions(cls, metric_names=None):
        """Get the default metric columns.

        Parameters
        ----------
        metric_names : list[str] | None
            List of metric names to get columns for. If None, all metrics are considered.

        Returns
        -------
        metric_column_descriptions : dict
            Dictionary of metric columns and their descriptions for each metric.
        """
        metric_column_descriptions = {}
        for m in cls.metric_list:
            if metric_names is not None and m.metric_name not in metric_names:
                continue
            if m.metric_descriptions is None:
                metric_column_descriptions.update({col: "no description" for col in m.metric_columns.keys()})
            else:
                if set(m.metric_descriptions.keys()) == set(m.metric_columns.keys()):
                    metric_column_descriptions.update(m.metric_descriptions)
                else:
                    warnings.warn(
                        f"Metric {m.metric_name} has inconsistent metric_descriptions and metric_columns keys."
                    )
        return metric_column_descriptions

    @classmethod
    def get_optional_dependencies(cls, **params):
        metric_names = params.get("metric_names", None)
        if metric_names is None:
            metric_names = [m.metric_name for m in cls.metric_list]
        else:
            for metric_name in metric_names:
                if metric_name not in [m.metric_name for m in cls.metric_list]:
                    raise ValueError(
                        f"Metric {metric_name} not in available metrics {[m.metric_name for m in cls.metric_list]}"
                    )
        metric_depend_on = set()
        for metric_name in metric_names:
            metric = [m for m in cls.metric_list if m.metric_name == metric_name][0]
            for dep in metric.depend_on:
                if "|" in dep:
                    dep_options = dep.split("|")
                    metric_depend_on.update(dep_options)
                else:
                    metric_depend_on.add(dep)
        depend_on = list(cls.depend_on) + list(metric_depend_on)
        return depend_on

    def _set_params(
        self,
        metric_names: list[str] | None = None,
        metric_params: dict | None = None,
        delete_existing_metrics: bool = False,
        metrics_to_compute: list[str] | None = None,
        periods: np.ndarray | None = None,
        **other_params,
    ):
        """
        Sets parameters for metric computation.

        Parameters
        ----------
        metric_names : list[str] | None
            List of metric names to compute. If None, all available metrics are computed.
        metric_params : dict | None
            Dictionary of metric parameters to override default parameters for specific metrics.
            If None, default parameters for all metrics are used.
        delete_existing_metrics : bool, default: False
            If True, existing metrics in the extension will be deleted before computing new ones.
        metrics_to_compute : list[str] | None
            List of metric names to compute. If None, all metrics in `metric_names` are computed.
        periods : np.ndarray | None
            Numpy array of unit_period_dtype defining periods to compute metrics over.
        other_params : dict
            Additional parameters for metric computation.

        Returns
        -------
        params : dict
            Dictionary of parameters for metric computation.

        Raises
        ------
        ValueError
            If any of the metric names are not in the available metrics.
        """
        # check metric names
        if metric_names is None:
            metric_names = [m.metric_name for m in self.metric_list]
        else:
            for metric_name in metric_names:
                if metric_name not in [m.metric_name for m in self.metric_list]:
                    raise ValueError(
                        f"Metric {metric_name} not in available metrics {[m.metric_name for m in self.metric_list]}"
                    )
        # check dependencies
        metrics_to_remove = []
        for metric_name in metric_names:
            metric = [m for m in self.metric_list if m.metric_name == metric_name][0]
            depend_on = metric.depend_on
            for dep in depend_on:
                if "|" in dep:
                    # at least one of the dependencies must be present
                    dep_options = dep.split("|")
                    if not any([self.sorting_analyzer.has_extension(d) for d in dep_options]):
                        metrics_to_remove.append(metric_name)
                else:
                    if not self.sorting_analyzer.has_extension(dep):
                        metrics_to_remove.append(metric_name)
            if metric.needs_recording and not self.sorting_analyzer.has_recording():
                warnings.warn(
                    f"Metric {metric_name} requires a recording. "
                    f"Since the SortingAnalyzer has no recording, the metric will not be computed."
                )
                metrics_to_remove.append(metric_name)

        metrics_to_remove = list(set(metrics_to_remove))
        if len(metrics_to_remove) > 0:
            warnings.warn(
                f"The following metrics will not be computed due to missing dependencies: {metrics_to_remove}"
            )

        for metric_name in metrics_to_remove:
            metric_names.remove(metric_name)

        default_metric_params = {m.metric_name: m.metric_params for m in self.metric_list}
        if metric_params is None:
            metric_params = default_metric_params
        else:
            for metric, params in metric_params.items():
                default_metric_params[metric].update(params)
            metric_params = default_metric_params

        if metrics_to_compute is None:
            metrics_to_compute = metric_names
        extension = self.sorting_analyzer.get_extension(self.extension_name)
        if delete_existing_metrics is False and extension is not None:
            existing_metric_names = extension.params["metric_names"]
            existing_metric_names_propagated = [
                metric_name for metric_name in existing_metric_names if metric_name not in metrics_to_compute
            ]
            metric_names = metrics_to_compute + existing_metric_names_propagated

        params = dict(
            metric_names=metric_names,
            metrics_to_compute=metrics_to_compute,
            delete_existing_metrics=delete_existing_metrics,
            metric_params=metric_params,
            periods=periods,
            **other_params,
        )
        return params

    def _prepare_data(self, sorting_analyzer, unit_ids=None):
        """
        Optional function to prepare shared data for metric computation.

        This function should return a dictionary containing any data that is shared across multiple metrics.
        The returned dictionary will be passed to each metric's compute function as `tmp_data` (if the metric
        requires it with the class attribute `needs_tmp_data=True`).
        """
        return {}

    def _compute_metrics(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids: list[int | str] | None = None,
        metric_names: list[str] | None = None,
        **job_kwargs,
    ):
        """
        Compute metrics.

        Parameters
        ----------
        sorting_analyzer : SortingAnalyzer
            The SortingAnalyzer object.
        unit_ids : list[int | str] | None, default: None
            List of unit ids to compute metrics for. If None, all units are used.
        metric_names : list[str] | None, default: None
            List of metric names to compute. If None, all metrics in params["metric_names"]
            are used.

        Returns
        -------
        metrics : pd.DataFrame
            DataFrame containing the computed metrics for each unit.
        run_times : dict
            Dictionary containing the computation time for each metric.

        """
        import pandas as pd

        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids
        tmp_data = self._prepare_data(sorting_analyzer=sorting_analyzer, unit_ids=unit_ids)
        if metric_names is None:
            metric_names = self.params["metric_names"]

        periods = self.params.get("periods", None)

        column_names_dtypes = {}
        for metric_name in metric_names:
            metric = [m for m in self.metric_list if m.metric_name == metric_name][0]
            column_names_dtypes.update(metric.metric_columns)

        metrics = pd.DataFrame(index=unit_ids, columns=list(column_names_dtypes.keys()))

        run_times = {}

        for metric_name in metric_names:
            metric = [m for m in self.metric_list if m.metric_name == metric_name][0]
            column_names = list(metric.metric_columns.keys())
            import time

            t_start = time.perf_counter()
            try:
                metric_params = self.params["metric_params"].get(metric_name, {})

                res = metric.compute(
                    sorting_analyzer,
                    unit_ids=unit_ids,
                    metric_params=metric_params,
                    tmp_data=tmp_data,
                    job_kwargs=job_kwargs,
                    periods=periods,
                )
            except Exception as e:
                warnings.warn(f"Error computing metric {metric_name}: {e}")
                if len(column_names) == 1:
                    res = {unit_id: np.nan for unit_id in unit_ids}
                else:
                    res = namedtuple("MetricResult", column_names)(*([np.nan] * len(column_names)))
            t_end = time.perf_counter()
            run_times[metric_name] = t_end - t_start

            # res is a namedtuple with several dictionary entries (one per column)
            if isinstance(res, dict):
                column_name = column_names[0]
                metrics.loc[unit_ids, column_name] = pd.Series(res)
            else:
                for i, col in enumerate(res._fields):
                    metrics.loc[unit_ids, col] = pd.Series(res[i])

        metrics = self._cast_metrics(metrics)

        return metrics, run_times

    def _run(self, **job_kwargs):

        metrics_to_compute = self.params["metrics_to_compute"]
        delete_existing_metrics = self.params["delete_existing_metrics"]
        periods = self.params.get("periods", None)

        _, job_kwargs = split_job_kwargs(job_kwargs)
        job_kwargs = fix_job_kwargs(job_kwargs)

        # compute the metrics which have been specified by the user
        computed_metrics, run_times = self._compute_metrics(
            sorting_analyzer=self.sorting_analyzer, unit_ids=None, metric_names=metrics_to_compute, **job_kwargs
        )

        existing_metrics = []

        # Check if we need to propagate any old metrics. If so, we'll do that.
        # Otherwise, we'll avoid attempting to load an empty metrics.
        if set(self.params["metrics_to_compute"]) != set(self.params["metric_names"]):

            extension = self.sorting_analyzer.get_extension(self.extension_name)
            if delete_existing_metrics is False and extension is not None and extension.data.get("metrics") is not None:
                existing_metrics = extension.params["metric_names"]

        existing_metrics = []
        # here we get in the loaded via the dict only (to avoid full loading from disk after params reset)
        extension = self.sorting_analyzer.extensions.get(self.extension_name, None)
        if delete_existing_metrics is False and extension is not None and extension.data.get("metrics") is not None:
            existing_metrics = extension.params["metric_names"]

        # append the metrics which were previously computed
        for metric_name in set(existing_metrics).difference(metrics_to_compute):
            metric = [m for m in self.metric_list if m.metric_name == metric_name][0]
            # some metrics names produce data columns with other names. This deals with that.
            for column_name in metric.metric_columns:
                computed_metrics[column_name] = extension.data["metrics"][column_name]

        self.data["metrics"] = computed_metrics
        self.data["runtime_s"] = run_times

    def _get_data(self):
        # convert to correct dtype
        return self.data["metrics"]

    def _cast_metrics(self, metrics_df):
        metric_dtypes = {}
        for m in self.metric_list:
            metric_dtypes.update(m.metric_columns)

        for col in metrics_df.columns:
            if col in metric_dtypes:
                try:
                    metrics_df[col] = metrics_df[col].astype(metric_dtypes[col])
                except Exception as e:
                    print(f"Error casting column {col}: {e}")
        return metrics_df

    def _select_extension_data(self, unit_ids: list[int | str]):
        """
        Select data for a subset of unit ids.

        Parameters
        ----------
        unit_ids : list[int | str]
            List of unit ids to select data for.

        Returns
        -------
        dict
            Dictionary containing the selected metrics DataFrame.
        """
        new_metrics = self.data["metrics"].loc[np.array(unit_ids)]
        return dict(metrics=new_metrics)

    def _merge_extension_data(
        self,
        merge_unit_groups: list[list[int | str]],
        new_unit_ids: list[int | str],
        new_sorting_analyzer: SortingAnalyzer,
        keep_mask: np.ndarray | None = None,
        verbose: bool = False,
        **job_kwargs,
    ):
        """
        Merge extension data from the old metrics DataFrame into the new one.

        Parameters
        ----------
        merge_unit_groups : list[list[int | str]]
            List of lists of unit ids to merge.
        new_unit_ids : list[int | str]
            List of new unit ids after merging.
        new_sorting_analyzer : SortingAnalyzer
            The new SortingAnalyzer object after merging.
        keep_mask : np.ndarray | None, default: None
            Mask to keep certain spikes (not used here).
        verbose : bool, default: False
            Whether to print verbose output.
        job_kwargs : dict
            Additional job keyword arguments.

        Returns
        -------
        dict
            Dictionary containing the merged metrics DataFrame.
        """
        import pandas as pd

        available_metric_names = [m.metric_name for m in self.metric_list]
        metric_names = [m for m in self.params["metric_names"] if m in available_metric_names]
        old_metrics = self.data["metrics"]

        all_unit_ids = new_sorting_analyzer.unit_ids
        not_new_ids = all_unit_ids[~np.isin(all_unit_ids, new_unit_ids)]

        metrics = pd.DataFrame(index=all_unit_ids, columns=old_metrics.columns)

        metrics.loc[not_new_ids, :] = old_metrics.loc[not_new_ids, :]
        metrics.loc[new_unit_ids, :], _ = self._compute_metrics(
            sorting_analyzer=new_sorting_analyzer, unit_ids=new_unit_ids, metric_names=metric_names, **job_kwargs
        )
        metrics = self._cast_metrics(metrics)

        new_data = dict(metrics=metrics)
        return new_data

    def _split_extension_data(
        self,
        split_units: dict[int | str, list[list[int]]],
        new_unit_ids: list[list[int | str]],
        new_sorting_analyzer: SortingAnalyzer,
        verbose: bool = False,
        **job_kwargs,
    ):
        """
        Split extension data from the old metrics DataFrame into the new one.

        Parameters
        ----------
        split_units : dict[int | str, list[list[int]]]
            List of unit ids to split.
        new_unit_ids : list[list[int | str]]
            List of lists of new unit ids after splitting.
        new_sorting_analyzer : SortingAnalyzer
            The new SortingAnalyzer object after splitting.
        verbose : bool, default: False
            Whether to print verbose output.
        """
        import pandas as pd
        from itertools import chain

        available_metric_names = [m.metric_name for m in self.metric_list]
        metric_names = [m for m in self.params["metric_names"] if m in available_metric_names]
        old_metrics = self.data["metrics"]

        all_unit_ids = new_sorting_analyzer.unit_ids
        new_unit_ids_f = list(chain(*new_unit_ids))
        not_new_ids = all_unit_ids[~np.isin(all_unit_ids, new_unit_ids_f)]

        metrics = pd.DataFrame(index=all_unit_ids, columns=old_metrics.columns)

        metrics.loc[not_new_ids, :] = old_metrics.loc[not_new_ids, :]
        metrics.loc[new_unit_ids_f, :], _ = self._compute_metrics(
            sorting_analyzer=new_sorting_analyzer, unit_ids=new_unit_ids_f, metric_names=metric_names, **job_kwargs
        )
        metrics = self._cast_metrics(metrics)

        new_data = dict(metrics=metrics)
        return new_data

    def set_data(self, ext_data_name, data):
        import pandas as pd

        if ext_data_name != "metrics":
            return
        if not isinstance(data, pd.DataFrame):
            return
        metrics = self._cast_metrics(data)
        self.data[ext_data_name] = metrics


class BaseSpikeVectorExtension(AnalyzerExtension):
    """
    Base class for spikevector-based extension, where the data is a numpy array with the same
    length as the spike vector.
    """

    extension_name = None  # to be defined in subclass
    need_recording = True
    use_nodepipeline = True
    need_job_kwargs = True
    need_backward_compatibility_on_load = False
    nodepipeline_variables = []  # to be defined in subclass

    def __init__(self, sorting_analyzer):
        super().__init__(sorting_analyzer)

    def _set_params(self, **kwargs):
        params = kwargs.copy()
        return params

    def _run(self, verbose=False, **job_kwargs):
        from spikeinterface.core.node_pipeline import run_node_pipeline

        # TODO: should we save directly to npy in binary_folder format / or to zarr?
        # if self.sorting_analyzer.format == "binary_folder":
        #     gather_mode = "npy"
        #     extension_folder = self.sorting_analyzer.folder / "extenstions" / self.extension_name
        #     gather_kwargs = {"folder": extension_folder}
        gather_mode = "memory"
        gather_kwargs = {}

        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        data = run_node_pipeline(
            self.sorting_analyzer.recording,
            nodes,
            job_kwargs=job_kwargs,
            job_name=self.extension_name,
            gather_mode=gather_mode,
            gather_kwargs=gather_kwargs,
            verbose=False,
        )
        if isinstance(data, tuple):
            # this logic enables extensions to optionally compute additional data based on params
            assert len(data) <= len(self.nodepipeline_variables), "Pipeline produced more outputs than expected"
        else:
            data = (data,)
        if len(self.nodepipeline_variables) > len(data):
            data_names = self.nodepipeline_variables[: len(data)]
        else:
            data_names = self.nodepipeline_variables
        for d, name in zip(data, data_names):
            self.data[name] = d

    def _get_data(self, outputs="numpy", concatenated=False, return_data_name=None, periods=None, copy=True):
        """
        Return extension data. If the extension computes more than one `nodepipeline_variables`,
        the `return_data_name` is used to specify which one to return.

        Parameters
        ----------
        outputs : "numpy" | "by_unit", default: "numpy"
            How to return the data, by default "numpy"
        concatenated : bool, default: False
            Whether to concatenate the data across segments.
        return_data_name : str | None, default: None
            The name of the data to return. If None and multiple `nodepipeline_variables` are computed,
            the first one is returned.
        periods : array of unit_period dtype, default: None
            Optional periods (segment_index, start_sample_index, end_sample_index, unit_index) to slice output data
        copy : bool, default: True
            Whether to return a copy of the data (only for outputs="numpy")

        Returns
        -------
        numpy.ndarray | dict
            The requested data in numpy or by unit format.
        """

        if len(self.nodepipeline_variables) == 1:
            return_data_name = self.nodepipeline_variables[0]
        else:
            if return_data_name is None:
                return_data_name = self.nodepipeline_variables[0]
            else:
                assert (
                    return_data_name in self.nodepipeline_variables
                ), f"return_data_name {return_data_name} not in nodepipeline_variables {self.nodepipeline_variables}"

        all_data = self.data[return_data_name]
        keep_mask = None
        if periods is not None:
            keep_mask = select_sorting_periods_mask(
                self.sorting_analyzer.sorting,
                periods,
            )
            all_data = all_data[keep_mask]
            # since we have the mask already, we can use it directly to avoid double computation
            spike_vector = self.sorting_analyzer.sorting.to_spike_vector(concatenated=True)
            sliced_spike_vector = spike_vector[keep_mask]
            sorting = NumpySorting(
                sliced_spike_vector,
                sampling_frequency=self.sorting_analyzer.sampling_frequency,
                unit_ids=self.sorting_analyzer.unit_ids,
            )
        else:
            sorting = self.sorting_analyzer.sorting

        if outputs == "numpy":
            if copy:
                return all_data.copy()  # return a copy to avoid modification
            else:
                return all_data
        elif outputs == "by_unit":
            unit_ids = self.sorting_analyzer.unit_ids

            if keep_mask is not None:
                # since we are filtering spikes, we need to recompute the spike indices
                spike_vector = sorting.to_spike_vector(concatenated=False)
                spike_indices = spike_vector_to_indices(spike_vector, unit_ids, absolute_index=True)
            else:
                # use the cache of indices
                spike_indices = self.sorting_analyzer.sorting.get_spike_vector_to_indices()
            data_by_units = {}
            for segment_index in range(self.sorting_analyzer.sorting.get_num_segments()):
                data_by_units[segment_index] = {}
                for unit_id in unit_ids:
                    inds = spike_indices[segment_index][unit_id]
                    data_by_units[segment_index][unit_id] = all_data[inds]

            if concatenated:
                data_by_units_concatenated = {
                    unit_id: np.concatenate([data_in_segment[unit_id] for data_in_segment in data_by_units.values()])
                    for unit_id in unit_ids
                }
                return data_by_units_concatenated

            return data_by_units
        else:
            raise ValueError(f"Wrong .get_data(outputs={outputs}); possibilities are `numpy` or `by_unit`")

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        for data_name in self.nodepipeline_variables:
            if self.data.get(data_name) is not None:
                new_data[data_name] = self.data[data_name][keep_spike_mask]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        new_data = dict()
        for data_name in self.nodepipeline_variables:
            if self.data.get(data_name) is not None:
                if keep_mask is None:
                    new_data[data_name] = self.data[data_name].copy()
                else:
                    new_data[data_name] = self.data[data_name][keep_mask]

        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # splitting only changes random spikes assignments
        return self.data.copy()
