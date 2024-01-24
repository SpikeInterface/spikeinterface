
from spikeinterface.core.sortingresult import register_result_extension, ResultExtension
from spikeinterface.core import get_noise_levels


class ComputeNoiseLevels(ResultExtension):
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
    sorting_result: SortingResult
        A SortingResult object
    **params: dict with additional parameters

    Returns
    -------
    noise_levels: np.array
        noise level vector.
    """
    extension_name = "noise_levels"

    def __init__(self, sorting_result):
        ResultExtension.__init__(self, sorting_result)

    def _set_params(self, num_chunks_per_segment=20, chunk_size=10000, return_scaled=True, seed=None):
        params = dict(num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, return_scaled=return_scaled, seed=seed)
        return params

    def _select_extension_data(self, unit_ids):
        # this do not depend on units
        return self.data

    def _run(self):
        self.data["noise_levels"] = get_noise_levels(self.sorting_result.recording,  **self.params)

    # def get_data(self):
    #     """
    #     Get computed noise levels.

    #     Returns
    #     -------
    #     noise_levels : np.array
    #         The noise levels associated to each channel.
    #     """
    #     return self._extension_data["noise_levels"]


register_result_extension(ComputeNoiseLevels)
compute_noise_levels = ComputeNoiseLevels.function_factory()
