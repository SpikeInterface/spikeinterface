"""
Implement AnalyzerExtension that are essential and imported in core
  * SelectRandomSpikes
  * ComputeWaveforms
  * ComputeTemplates
Theses two classes replace the WaveformExtractor

It also implement:
  * ComputeFastTemplates which is equivalent but without extacting waveforms.
  * ComputeNoiseLevels which is very convenient to have
"""

import numpy as np

from .sortinganalyzer import AnalyzerExtension, register_result_extension
from .waveform_tools import extract_waveforms_to_single_buffer, estimate_templates_average
from .recording_tools import get_noise_levels
from .template import Templates
from .sorting_tools import random_spikes_selection


class SelectRandomSpikes(AnalyzerExtension):
    """
    AnalyzerExtension that select some random spikes.

    This will be used by "compute_waveforms" and so "compute_templates" or "compute_fast_templates"

    This internally use `random_spikes_selection()` parameters are the same.

    Parameters
    ----------
    unit_ids: list or None
        Unit ids to retrieve waveforms for
    mode: "average" | "median" | "std" | "percentile", default: "average"
        The mode to compute the templates
    percentile: float, default: None
        Percentile to use for mode="percentile"
    save: bool, default True
        In case, the operator is not computed yet it can be saved to folder or zarr.

    Returns
    -------

    """

    extension_name = "random_spikes"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _run(
        self,
    ):
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

    def _get_data(self):
        return self.data["random_spikes_indices"]

    def some_spikes(self):
        # utils to get the some_spikes vector
        # use internal cache
        if not hasattr(self, "_some_spikes"):
            spikes = self.sorting_analyzer.sorting.to_spike_vector()
            self._some_spikes = spikes[self.data["random_spikes_indices"]]
        return self._some_spikes

    def get_selected_indices_in_spike_train(self, unit_id, segment_index):
        # usefull for Waveforms extractor backwars compatibility
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


compute_select_random_spikes = SelectRandomSpikes.function_factory()
register_result_extension(SelectRandomSpikes)


class ComputeWaveforms(AnalyzerExtension):
    """
    AnalyzerExtension that extract some waveforms of each units.

    The sparsity is controlled by the SortingAnalyzer sparsity.
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

    def _run(self, **job_kwargs):
        self.data.clear()

        # if self.sorting_analyzer.random_spikes_indices is None:
        #     raise ValueError("compute_waveforms need SortingAnalyzer.select_random_spikes() need to be run first")

        # random_spikes_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        # spikes = sorting.to_spike_vector()
        # some_spikes = spikes[random_spikes_indices]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").some_spikes()

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
            return_scaled=self.params["return_scaled"],
            file_path=file_path,
            dtype=self.params["dtype"],
            sparsity_mask=sparsity_mask,
            copy=copy,
            job_name="compute_waveforms",
            **job_kwargs,
        )

        self.data["waveforms"] = all_waveforms

    def _set_params(
        self,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        return_scaled: bool = True,
        dtype=None,
    ):
        recording = self.sorting_analyzer.recording
        if dtype is None:
            dtype = recording.get_dtype()

        if return_scaled:
            # check if has scaled values:
            if not recording.has_scaled() and recording.get_dtype().kind == "i":
                print("Setting 'return_scaled' to False")
                return_scaled = False

        if np.issubdtype(dtype, np.integer) and return_scaled:
            dtype = "float32"

        dtype = np.dtype(dtype)

        params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            return_scaled=return_scaled,
            dtype=dtype.str,
        )
        return params

    def _select_extension_data(self, unit_ids):
        # random_spikes_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").some_spikes()

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))
        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        # some_spikes = spikes[random_spikes_indices]
        keep_spike_mask = np.isin(some_spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["waveforms"] = self.data["waveforms"][keep_spike_mask, :, :]

        return new_data

    def get_waveforms_one_unit(
        self,
        unit_id,
        force_dense: bool = False,
    ):
        sorting = self.sorting_analyzer.sorting
        unit_index = sorting.id_to_index(unit_id)
        # spikes = sorting.to_spike_vector()
        # some_spikes = spikes[self.sorting_analyzer.random_spikes_indices]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").some_spikes()
        spike_mask = some_spikes["unit_index"] == unit_index
        wfs = self.data["waveforms"][spike_mask, :, :]

        if self.sorting_analyzer.sparsity is not None:
            chan_inds = self.sorting_analyzer.sparsity.unit_id_to_channel_indices[unit_id]
            wfs = wfs[:, :, : chan_inds.size]
            if force_dense:
                num_channels = self.get_num_channels()
                dense_wfs = np.zeros((wfs.shape[0], wfs.shape[1], num_channels), dtype=wfs.dtype)
                dense_wfs[:, :, chan_inds] = wfs
                wfs = dense_wfs

        return wfs

    def _get_data(self):
        return self.data["waveforms"]


compute_waveforms = ComputeWaveforms.function_factory()
register_result_extension(ComputeWaveforms)


class ComputeTemplates(AnalyzerExtension):
    """
    AnalyzerExtension that compute templates (average, str, median, percentile, ...)

    This must be run after "waveforms" extension (`SortingAnalyzer.compute("waveforms")`)

    Note that when "waveforms" is already done, then the recording is not needed anymore for this extension.

    Note: by default only the average is computed. Other operator (std, median, percentile) can be computed on demand
    after the SortingAnalyzer.compute("templates") and then the data dict is updated on demand.


    """

    extension_name = "templates"
    depend_on = ["waveforms"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _set_params(self, operators=["average", "std"]):
        assert isinstance(operators, list)
        for operator in operators:
            if isinstance(operator, str):
                assert operator in ("average", "std", "median", "mad")
            else:
                assert isinstance(operator, (list, tuple))
                assert len(operator) == 2
                assert operator[0] == "percentile"

        waveforms_extension = self.sorting_analyzer.get_extension("waveforms")

        params = dict(
            operators=operators,
            nbefore=waveforms_extension.nbefore,
            nafter=waveforms_extension.nafter,
            return_scaled=waveforms_extension.params["return_scaled"],
        )
        return params

    def _run(self):
        self._compute_and_append(self.params["operators"])

    def _compute_and_append(self, operators):
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
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").some_spikes()
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
        return self.params["nbefore"]

    @property
    def nafter(self):
        return self.params["nafter"]

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        new_data = dict()
        for key, arr in self.data.items():
            new_data[key] = arr[keep_unit_indices, :, :]

        return new_data

    def _get_data(self, operator="average", percentile=None, outputs="numpy"):
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=..."
            key = f"pencentile_{percentile}"

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
            raise ValueError("outputs must be numpy or Templates")

    def get_templates(self, unit_ids=None, operator="average", percentile=None, save=True):
        """
        Return templates (average, std, median or percentil) for multiple units.

        I not computed yet then this is computed on demand and optionally saved.

        Parameters
        ----------
        unit_ids: list or None
            Unit ids to retrieve waveforms for
        mode: "average" | "median" | "std" | "percentile", default: "average"
            The mode to compute the templates
        percentile: float, default: None
            Percentile to use for mode="percentile"
        save: bool, default True
            In case, the operator is not computed yet it can be saved to folder or zarr.

        Returns
        -------
        templates: np.array
            The returned templates (num_units, num_samples, num_channels)
        """
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=..."
            key = f"pencentile_{percentile}"

        if key in self.data:
            templates = self.data[key]
        else:
            if operator != "percentile":
                self._compute_and_append([operator])
                self.params["operators"] += [operator]
            else:
                self._compute_and_append([(operator, percentile)])
                self.params["operators"] += [(operator, percentile)]
            templates = self.data[key]

        if save:
            self.save()

        if unit_ids is not None:
            unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
            templates = templates[unit_indices, :, :]

        return np.array(templates)


compute_templates = ComputeTemplates.function_factory()
register_result_extension(ComputeTemplates)


class ComputeFastTemplates(AnalyzerExtension):
    """
    AnalyzerExtension which is similar to the extension "templates" (ComputeTemplates) **but only for average**.
    This is way faster because it do not need "waveforms" to be computed first.
    """

    extension_name = "fast_templates"
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

    def _run(self, **job_kwargs):
        self.data.clear()

        # if self.sorting_analyzer.random_spikes_indices is None:
        #     raise ValueError("compute_waveforms need SortingAnalyzer.select_random_spikes() need to be run first")

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        # spikes = sorting.to_spike_vector()
        # some_spikes = spikes[self.sorting_analyzer.random_spikes_indices]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").some_spikes()

        return_scaled = self.params["return_scaled"]

        # TODO jobw_kwargs
        self.data["average"] = estimate_templates_average(
            recording, some_spikes, unit_ids, self.nbefore, self.nafter, return_scaled=return_scaled, **job_kwargs
        )

    def _set_params(
        self,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        return_scaled: bool = True,
    ):
        params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            return_scaled=return_scaled,
        )
        return params

    def _get_data(self, outputs="numpy"):
        templates_array = self.data["average"]

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
            raise ValueError("outputs must be numpy or Templates")

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        new_data = dict()
        new_data["average"] = self.data["average"][keep_unit_indices, :, :]

        return new_data


compute_fast_templates = ComputeFastTemplates.function_factory()
register_result_extension(ComputeFastTemplates)


class ComputeNoiseLevels(AnalyzerExtension):
    """
    Computes the noise level associated to each recording channel.

    This function will wraps the `get_noise_levels(recording)` to make the noise levels persistent
    on disk (folder or zarr) as a `WaveformExtension`.
    The noise levels do not depend on the unit list, only the recording, but it is a convenient way to
    retrieve the noise levels directly ine the WaveformExtractor.

    Note that the noise levels can be scaled or not, depending on the `return_scaled` parameter
    of the `WaveformExtractor`.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    **params: dict with additional parameters

    Returns
    -------
    noise_levels: np.array
        noise level vector.
    """

    extension_name = "noise_levels"
    depend_on = []
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = False

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(self, num_chunks_per_segment=20, chunk_size=10000, return_scaled=True, seed=None):
        params = dict(
            num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, return_scaled=return_scaled, seed=seed
        )
        return params

    def _select_extension_data(self, unit_ids):
        # this do not depend on units
        return self.data

    def _run(self):
        self.data["noise_levels"] = get_noise_levels(self.sorting_analyzer.recording, **self.params)

    def _get_data(self):
        return self.data["noise_levels"]


register_result_extension(ComputeNoiseLevels)
compute_noise_levels = ComputeNoiseLevels.function_factory()
