from __future__ import annotations

from spikeinterface.core.waveform_extractor import BaseWaveformExtractorExtension, WaveformExtractor
from spikeinterface.core import get_noise_levels


class NoiseLevelsCalculator(BaseWaveformExtractorExtension):
    extension_name = "noise_levels"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

    def _set_params(self, num_chunks_per_segment=20, chunk_size=10000, seed=None):
        params = dict(num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, seed=seed)
        return params

    def _select_extension_data(self, unit_ids):
        # this do not depend on units
        return self._extension_data

    def _run(self):
        return_scaled = self.waveform_extractor.return_scaled
        self._extension_data["noise_levels"] = get_noise_levels(
            self.waveform_extractor.recording, return_scaled=return_scaled, **self._params
        )

    def get_data(self):
        """
        Get computed noise levels.

        Returns
        -------
        noise_levels : np.array
            The noise levels associated to each channel.
        """
        return self._extension_data["noise_levels"]

    @staticmethod
    def get_extension_function():
        return compute_noise_levels


WaveformExtractor.register_extension(NoiseLevelsCalculator)


def compute_noise_levels(waveform_extractor, load_if_exists=False, **params):
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
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    load_if_exists: bool, default: False
        If True, the noise levels are loaded if they already exist
    **params: dict with additional parameters


    Returns
    -------
    noise_levels: np.array
        noise level vector.
    """
    if load_if_exists and waveform_extractor.is_extension(NoiseLevelsCalculator.extension_name):
        ext = waveform_extractor.load_extension(NoiseLevelsCalculator.extension_name)
    else:
        ext = NoiseLevelsCalculator(waveform_extractor)
        ext.set_params(**params)
        ext.run()

    return ext.get_data()
