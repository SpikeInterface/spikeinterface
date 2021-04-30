from scipy.signal import convolve
from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args

from spiketoolkit.preprocessing.basepreprocessorrecording import (
    BasePreprocessorRecordingExtractor,
)
import numpy as np


class ArtifactSuppressionRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = "Clip"
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(
        self,
        recording,
        thresh=2000,
        ms_surrounding=50,
        fill_mode="zero",
        noise_fill_std=3,
    ):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._thresh = thresh
        self._ms_surrounding = ms_surrounding
        self._fill_mode = fill_mode
        self._noise_fill_std = noise_fill_std
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self.has_unscaled = False
        self._kwargs = {"recording": recording.make_serialized_dict(), "thresh": thresh}

    @check_get_traces_args
    def get_traces(
        self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True
    ):
        assert return_scaled, "'clip' only supports return_scaled=True"

        traces = self._recording.get_traces(
            channel_ids=channel_ids,
            start_frame=start_frame,
            end_frame=end_frame,
            return_scaled=return_scaled,
        )

        frames_surrounding = int(
            self._ms_surrounding * self.get_sampling_frequency() / 1000
        )
        above_thresh = np.abs(traces) > self._thresh
        # convolve with surrounding frames
        above_thresh = (
            convolve(above_thresh, np.ones((1, int(frames_surrounding))), mode="same")
            > 1e-10
        )

        if self._fill_mode == "noise":
            trace_stds = np.array(
                [
                    np.std(trace[thresh == False])
                    for trace, thresh in zip(traces, above_thresh)
                ]
            )
            trace_means = np.array(
                [
                    np.mean(trace[thresh == False])
                    for trace, thresh in zip(traces, above_thresh)
                ]
            )
            noise = (
                (
                    np.random.rand(np.product(np.shape(traces))).reshape(
                        np.shape(traces)[::-1]
                    )
                    - 0.5
                )
                * np.expand_dims(trace_stds, 0)
                * self._noise_fill_std
                + np.expand_dims(trace_means, 0)
            ).T
            traces[above_thresh] = noise[above_thresh]
        else:
            traces[above_thresh] = 0
        return traces


def suppress_artifacts(
    recording, thresh=2000, ms_surrounding=50, fill_mode="zero", noise_fill_std=3
):
    """
    Finds and removes artifacts by thresholding voltage and replacing the time around
    that artifact with either zeros, or noise. This is meant to be applied after 
    bandpass filtering and common average referencing. 
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    thresh: float  (default 2000)
        Threshold for determining noise artifact
    ms_surrounding: float  (default 50)
        Surrounding milliseconds from artifact to remove
    fill_mode: str (default 'zero')
        Fill removed artifacts with either 0s ('zero') or uniform noise ('noise')
    noise_fill_std: float (default 3)
        Number of standard deviations the noise should be if fill_mode is 'noise'
    Returns
    -------
    rescaled_traces: ClipTracesRecording
        The clipped traces recording extractor object
    """
    return ArtifactSuppressionRecording(
        recording=recording,
        thresh=thresh,
        ms_surrounding=ms_surrounding,
        fill_mode=fill_mode,
        noise_fill_std=noise_fill_std,
    )
