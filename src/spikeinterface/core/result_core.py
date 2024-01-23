"""
Implement ResultExtension that are essential and imported in core
  * ComputeWaveforms
  * ComputeTemplates
Theses two classes replace the WaveformExtractor

It also implement ComputeFastTemplates which is equivalent but without extacting waveforms.
"""

import numpy as np

from .sortingresult import ResultExtension, register_result_extension
from .waveform_tools import extract_waveforms_to_single_buffer, estimate_templates

class ComputeWaveforms(ResultExtension):
    """
    ResultExtension that extract some waveforms of each units.

    The sparsity is controlled by the SortingResult sparsity.
    """
    extension_name = "waveforms"
    depend_on = []
    need_recording = True
    use_nodepipeline = False

    @property
    def nbefore(self):
        return int(self.params["ms_before"] * self.sorting_result.sampling_frequency / 1000.0)

    @property
    def nafter(self):
        return int(self.params["ms_after"] * self.sorting_result.sampling_frequency / 1000.0)

    def _run(self, **kwargs):
        self.data.clear()

        if self.sorting_result.random_spikes_indices is None:
            raise ValueError("compute_waveforms need SortingResult.select_random_spikes() need to be run first")

        recording = self.sorting_result.recording
        sorting = self.sorting_result.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        spikes = sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        
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

        # TODO propagate some job_kwargs
        job_kwargs = dict(n_jobs=-1)

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

    def _set_params(self, 
            ms_before: float = 1.0,
            ms_after: float = 2.0,
            return_scaled: bool = True,
            dtype=None,
        ):
        recording = self.sorting_result.recording
        if dtype is None:
            dtype = recording.get_dtype()

        if return_scaled:
            # check if has scaled values:
            if not recording.has_scaled():
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
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))
        spikes = self.sorting_result.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        keep_spike_mask = np.isin(some_spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["waveforms"] = self.data["waveforms"][keep_spike_mask, :, :]

        return new_data




compute_waveforms = ComputeWaveforms.function_factory()
register_result_extension(ComputeWaveforms)


class ComputeTemplates(ResultExtension):
    """
    ResultExtension that compute templates (average, str, median, percentile, ...)
    
    This must be run after "waveforms" extension (`SortingResult.compute("waveforms")`)

    Note that when "waveforms" is already done, then the recording is not needed anymore for this extension.
    """
    extension_name = "templates"
    depend_on = ["waveforms"]
    need_recording = False
    use_nodepipeline = False

    def _run(self, **kwargs):
        
        unit_ids = self.sorting_result.unit_ids
        channel_ids = self.sorting_result.channel_ids
        waveforms_extension = self.sorting_result.get_extension("waveforms")
        waveforms = waveforms_extension.data["waveforms"]
        
        num_samples = waveforms.shape[1]
        
        for operator in self.params["operators"]:
            if isinstance(operator, str) and operator in ("average", "std", "median"):
                key = operator
            elif isinstance(operator, (list, tuple)):
                operator, percentile = operator
                assert operator == "percentile"
                key = f"pencentile_{percentile}"
            else:
                raise ValueError(f"ComputeTemplates: wrong operator {operator}")
            self.data[key] = np.zeros((unit_ids.size, num_samples, channel_ids.size))

        spikes = self.sorting_result.sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        for unit_index, unit_id in enumerate(unit_ids):
            spike_mask = some_spikes["unit_index"] == unit_index
            wfs = waveforms[spike_mask, :, :]
            if wfs.shape[0] == 0:
                continue
            
            for operator in self.params["operators"]:
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
                    self.data[key][unit_index, :, :][:, channel_indices] = arr[:, :channel_indices.size]

    def _set_params(self, operators = ["average", "std"]):
        assert isinstance(operators, list)
        for operator in operators:
            if isinstance(operator, str):
                assert operator in ("average", "std", "median", "mad")
            else:
                assert isinstance(operator, (list, tuple))
                assert len(operator) == 2
                assert operator[0] == "percentile"

        waveforms_extension = self.sorting_result.get_extension("waveforms")

        params = dict(operators=operators, nbefore=waveforms_extension.nbefore, nafter=waveforms_extension.nafter)
        return params

    @property
    def nbefore(self):
        return self.params["nbefore"]

    @property
    def nafter(self):
        return self.params["nafter"]

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))

        new_data = dict()
        for key, arr in self.data.items():
            new_data[key] = arr[keep_unit_indices, :, :]

        return new_data

compute_templates = ComputeTemplates.function_factory()
register_result_extension(ComputeTemplates)


class ComputeFastTemplates(ResultExtension):
    """
    ResultExtension which is similar to the extension "templates" (ComputeTemplates) **but only for average**.
    This is way faster because it do not need "waveforms" to be computed first. 
    """
    extension_name = "fast_templates"
    depend_on = []
    need_recording = True
    use_nodepipeline = False

    @property
    def nbefore(self):
        return int(self.params["ms_before"] * self.sorting_result.sampling_frequency / 1000.0)

    @property
    def nafter(self):
        return int(self.params["ms_after"] * self.sorting_result.sampling_frequency / 1000.0)

    def _run(self, **kwargs):
        self.data.clear()

        if self.sorting_result.random_spikes_indices is None:
            raise ValueError("compute_waveforms need SortingResult.select_random_spikes() need to be run first")

        recording = self.sorting_result.recording
        sorting = self.sorting_result.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        spikes = sorting.to_spike_vector()
        some_spikes = spikes[self.sorting_result.random_spikes_indices]
        
        return_scaled = self.params["return_scaled"]

        # TODO jobw_kwargs
        self.data["average"] = estimate_templates(recording, some_spikes, unit_ids, self.nbefore, self.nafter, return_scaled=return_scaled)
    
    def _set_params(self,
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

        

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))

        new_data = dict()
        new_data["average"] = self.data["average"][keep_unit_indices, :, :]

        return new_data

compute_fast_templates = ComputeFastTemplates.function_factory()
register_result_extension(ComputeFastTemplates)
