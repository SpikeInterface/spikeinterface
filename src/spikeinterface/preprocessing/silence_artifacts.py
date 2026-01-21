from __future__ import annotations

import numpy as np

from spikeinterface.core.base import base_peak_dtype
from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.core.node_pipeline import PeakDetector
from spikeinterface.preprocessing.silence_periods import SilencedPeriodsRecording
import numpy as np


# class SilencedArtifactsRecording(SilencedPeriodsRecording):
#     """
#     Silence user-defined periods from recording extractor traces. The code will construct
#     an enveloppe of the recording (as a low pass filtered version of the traces) and detect
#     threshold crossings to identify the periods to silence. The periods are then silenced either
#     on a per channel basis or across all channels by replacing the values by zeros or by
#     adding gaussian noise with the same variance as the one in the recordings

#     Parameters
#     ----------
#     recording : RecordingExtractor
#         The recording extractor to silence putative artifacts
#     artifacts : np.array, None
#         The threshold to detect artifacts. The threshold is computed as `detect_threshold * noise_level`
#     freq_max : float, default: 20
#         The maximum frequency for the low pass filter used
#     min_duration_ms : float, default: 50
#         The minimum duration for a threshold crossing to be considered as an artefact.
#     noise_levels : array
#         Noise levels if already computed
#     seed : int | None, default: None
#         Random seed for `get_noise_levels` and `NoiseGeneratorRecording`.
#         If none, `get_noise_levels` uses `seed=0` and `NoiseGeneratorRecording` generates a random seed using `numpy.random.default_rng`.
#     mode : "zeros" | "noise", default: "zeros"
#         Determines what periods are replaced by. Can be one of the following:

#         - "zeros": Artifacts are replaced by zeros.

#         - "noise": The periods are filled with a gaussion noise that has the
#                    same variance that the one in the recordings, on a per channel
#                    basis
#     **noise_levels_kwargs : Keyword arguments for `spikeinterface.core.get_noise_levels()` function

#     Returns
#     -------
#     silenced_recording : SilencedArtifactsRecording
#         The recording extractor after silencing detected artifacts
#     """

#     _precomputable_kwarg_names = ["artifacts"]

#     def __init__(
#         self,
#         recording,
#         artifacts=None,
#         detect_threshold=5,
#         verbose=False,
#         freq_max=20.0,
#         min_duration_ms=50,
#         mode="zeros",
#         noise_levels=None,
#         seed=None,
#         list_periods=None,
#         **noise_levels_kwargs,
#     ):

#         if artifacts is None:
#             from spikeinterface.preprocessing import detect_artifacts
#             artifacts = detect_artifact_periods(
#                 recording,
#                 detect_threshold=detect_threshold,
#                 min_duration_ms=min_duration_ms,
#                 freq_max=freq_max,
#                 seed=seed,
#                 noise_levels=noise_levels,
#                 **noise_levels_kwargs,
#             )

#             if verbose:
#                 for i, periods in enumerate(artifacts):
#                     total_time = np.sum([end - start for start, end in periods])
#                     percentage = 100 * total_time / recording.get_num_samples(i)
#                     print(f"{percentage}% of segment {i} has been flagged as artifactual")

#         SilencedPeriodsRecording.__init__(
#             self, recording, artifacts, mode=mode, noise_levels=noise_levels, seed=seed, **noise_levels_kwargs
#         )


# # function for API
# silence_artifacts = define_function_handling_dict_from_class(
#     source_class=SilencedArtifactsRecording, name="silence_artifacts"
# )
