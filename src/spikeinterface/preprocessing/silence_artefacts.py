from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_from_class
from .silence_periods import SilencedPeriodsRecording
from .rectify import RectifyRecording
from .filter_gaussian import GaussianFilterRecording
from ..core.job_tools import split_job_kwargs, fix_job_kwargs

from ..core import  get_noise_levels
from ..core.generate import NoiseGeneratorRecording
from .basepreprocessor import BasePreprocessor


from ..core.node_pipeline import PeakDetector, base_peak_dtype
import numpy as np

class DetectThresholdCrossing(PeakDetector):
    
    name = "threshold_crossings"
    preferred_mp_context = None
    
    def __init__(
        self,
        recording,
        detect_threshold=5,
        noise_levels=None,
        random_chunk_kwargs={},
    ):
        PeakDetector.__init__(self, recording, return_output=True)
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        self.abs_thresholds = noise_levels * detect_threshold
        self._dtype = np.dtype(base_peak_dtype + [("onset", "bool")])

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype
    
    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        z =  (traces - self.abs_thresholds).mean(1)
        threshold_mask = np.diff((z > 0) != 0, axis=0)
        indices,  = np.where(threshold_mask)
        local_peaks = np.zeros(indices.size, dtype=self._dtype)
        local_peaks["sample_index"] = indices
        #local_peaks["channel_index"] = indices[-1]
        #for channel_ind in np.unique(indices[1]):
        #    mask = np.flatnonzero(indices[1] == channel_ind)
        #    local_peaks["onset"][mask[::2]] = True
        #    local_peaks["onset"][mask[1::2]] = False
        #idx = np.argsort(local_peaks["sample_index"])
        #local_peaks = local_peaks[idx]
        local_peaks["onset"][::2] = True
        local_peaks["onset"][1::2] = False
        return (local_peaks, )


def detect_onsets(recording, detect_threshold=5, **job_kwargs):

    from spikeinterface.core.node_pipeline import (
        run_node_pipeline,
    )

    node0 = DetectThresholdCrossing(recording, detect_threshold, **job_kwargs)
    
    peaks = run_node_pipeline(
        recording,
        [node0],
        job_kwargs,
        job_name="detecting threshold crossings",
        )

    results = []
    print(peaks)
    # for channel_ind, channel_id in enumerate(recording.channel_ids):

    #     mask = peaks["channel_index"] == channel_ind
    #     sub_peaks = peaks[mask]
    #     onset_mask = sub_peaks["onset"] == 1
    #     onsets = sub_peaks[onset_mask]
    #     offsets = sub_peaks[~onset_mask]
    #     periods = []

    #     # if len(onsets) > 0:
    #     #     if onsets['sample_index'][0] > offsets['sample_index'][0]:
    #     #         periods += [(0, offsets['sample_index'][0])]
    #     #         offsets = offsets[1:]
                
    #     #     for i in range(len(onsets)):
    #     #         periods += [(onsets['sample_index'][i], offsets['sample_index'][i])]
            
    #     #     if len(onsets) > len(offsets):
    #     #        periods += [(onsets['sample_index'][0], recording.get_num_samples())]
                
    #     results[channel_id] = periods
        
    return results


class SilencedArtefactsRecording(SilencedPeriodsRecording):
    """
    Silence user-defined periods from recording extractor traces. The code will construct
    an enveloppe of the recording (as a low pass filtered version of the traces) and detect
    threshold crossings to identify the periods to silence. The periods are then silenced either
    on a per channel basis or across all channels by replacing the values by zeros or by 
    adding gaussian noise with the same variance as the one in the recordings

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to silence putative artefacts
    detect_threshold : float, default: 5
        The threshold to detect artefacts. The threshold is computed as `detect_threshold * noise_level`
    freq_max : float, default: 20
        The maximum frequency for the low pass filter used
    noise_levels : array
        Noise levels if already computed
    seed : int | None, default: None
        Random seed for `get_noise_levels` and `NoiseGeneratorRecording`.
        If none, `get_noise_levels` uses `seed=0` and `NoiseGeneratorRecording` generates a random seed using `numpy.random.default_rng`.
    mode : "zeros" | "noise, default: "zeros"
        Determines what periods are replaced by. Can be one of the following:

        - "zeros": Artifacts are replaced by zeros.

        - "noise": The periods are filled with a gaussion noise that has the
                   same variance that the one in the recordings, on a per channel
                   basis
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    silenced_recording : SilencedArtefactsRecording
        The recording extractor after silencing detected artefacts
    """

    def __init__(self, 
                 recording,
                 per_channel=True,
                 detect_threshold=5,
                 freq_max=20., 
                 mode="zeros", 
                 noise_levels=None,
                 seed=None, 
                 **random_chunk_kwargs):
        

        _, job_kwargs  = split_job_kwargs(random_chunk_kwargs)
        job_kwargs = fix_job_kwargs(job_kwargs)

        recording = RectifyRecording(recording)
        recording = GaussianFilterRecording(recording, freq_min=None, freq_max=freq_max)

        periods = detect_onsets(recording, 
                                detect_threshold=detect_threshold, 
                                **random_chunk_kwargs)

        SilencedPeriodsRecording.__init__(self, 
                                          recording, 
                                          periods, 
                                          mode=mode, 
                                          noise_levels=noise_levels, 
                                          seed=seed, 
                                          **random_chunk_kwargs)


# function for API
silence_artefacts = define_function_from_class(source_class=SilencedArtefactsRecording, name="silence_artefacts")
