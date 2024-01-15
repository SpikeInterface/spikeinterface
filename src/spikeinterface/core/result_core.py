"""
Implement ResultExtension that are essential and imported in core
  * ComputeWaveforms
  * ComputTemplates

Theses two classes replace the WaveformExtractor

"""

import numpy as np

from .sortingresult import ResultExtension, register_result_extension
from .waveform_tools import extract_waveforms_to_single_buffer

class ComputeWaveforms(ResultExtension):
    extension_name = "waveforms"
    depend_on = []
    need_recording = True
    use_nodepiepline = False

    def _run(self, **kwargs):
        self.data.clear()

        if self.sorting_result.random_spikes_indices is None:
            raise ValueError("compute_waveforms need SortingResult.select_random_spikes() need to be run first")



        recording = self.sorting_result.recording
        sorting = self.sorting_result.sorting
        # TODO handle sampling
        spikes = sorting.to_spike_vector()
        unit_ids = sorting.unit_ids

        nbefore = int(self.params["ms_before"] * sorting.sampling_frequency / 1000.0)
        nafter = int(self.params["ms_after"] * sorting.sampling_frequency / 1000.0)


        # TODO find a solution maybe using node pipeline : here the waveforms is directly written to memamap 
        # the save will delete the memmap write it again!! this will not work on window.
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

        some_spikes = spikes[self.sorting_result.random_spikes_indices]

        all_waveforms = extract_waveforms_to_single_buffer(
            recording,
            some_spikes,
            unit_ids,
            nbefore,
            nafter,
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
            max_spikes_per_unit: int = 500,
            return_scaled: bool = False,
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

        if max_spikes_per_unit is not None:
            max_spikes_per_unit = int(max_spikes_per_unit)

        params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=max_spikes_per_unit,
            return_scaled=return_scaled,
            dtype=dtype.str,
        )
        return params

    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

        # keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))
        # spikes = self.sorting_result.sorting.to_spike_vector()
        # keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)




compute_waveforms = ComputeWaveforms.function_factory()
register_result_extension(ComputeWaveforms)

class ComputTemplates(ResultExtension):
    extension_name = "templates"
    depend_on = ["waveforms"]
    need_recording = False
    use_nodepiepline = False

    def _run(self, **kwargs):
        # must be implemented in subclass
        # must populate the self.data dictionary
        raise NotImplementedError

    def _set_params(self, **params):
        # must be implemented in subclass
        # must return a cleaned version of params dict
        raise NotImplementedError

    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

compute_templates = ComputTemplates.function_factory()
register_result_extension(ComputTemplates)