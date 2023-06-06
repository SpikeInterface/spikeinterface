# Name will be chosen when reviewing
from pathlib import Path

from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms, run_node_pipeline
from spikeinterface.sortingcomponents.peak_detection import IterativePeakDetector
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCADenoising, TemporalPCAProjection
from spikeinterface.sortingcomponents.peak_detection import (
    DetectPeakLocallyExclusive,
)


def train_pca(recording, job_kwargs, tmp_path):
    # Temporary function to be removed when structure is discussed

    ms_before = 1.0
    ms_after = 1.0

    model_folder_path = Path(tmp_path) / "temporal_pca_model"
    if not model_folder_path.is_dir():
        model_folder_path.mkdir(parents=True, exist_ok=True)
        # Fit the model
        n_components = 3
        n_peaks = 100  # Heuristic for extracting around 1k waveforms per channel
        peak_selection_params = dict(method="uniform", select_per_channel=True, n_peaks=n_peaks)
        detect_peaks_params = dict(method="by_channel", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1)
        TemporalPCADenoising.fit(
            recording=recording,
            model_folder_path=model_folder_path,
            n_components=n_components,
            ms_before=ms_before,
            ms_after=ms_after,
            detect_peaks_params=detect_peaks_params,
            peak_selection_params=peak_selection_params,
            job_kwargs=job_kwargs,
        )

    return model_folder_path


class ColumbiaSorter2022:
    def __init__(self, recording: BaseRecording, job_kwargs) -> None:
        # Load all the parameters and do checks

        # What do we need to do here?
        # 1 Check if the recording is already preprocessed
        # 2 check if the recording is already filtered?

        # We need a peak detector
        # Then we need to extract features for clustering which are going to be
        # Positions and TemporalPCA features.

        # Then template matching to find spikes from templates
        self.job_kwargs = job_kwargs
        self.recording = recording

        peak_detector_kwargs = dict(
            recording=recording,
            exclude_sweep_ms=1.0,
            peak_sign="both",
            detect_threshold=5,
            local_radius_um=50,
        )

        peak_detector_node = DetectPeakLocallyExclusive(**peak_detector_kwargs)

        ms_before = 1.0
        ms_after = 1.0
        self.waveform_extraction_node = ExtractDenseWaveforms(
            recording=recording, ms_before=ms_before, ms_after=ms_after
        )

        # Create a temporary path
        import tempfile

        # Create a temporary directory
        # tmp_path = tempfile.mkdtemp()
        tmp_path = Path("/home/heberto/tmp_for_sorting_pipeline/")
        self.pca_model_folder_path = train_pca(recording, job_kwargs=job_kwargs, tmp_path=tmp_path)
        waveform_denoising_node = TemporalPCADenoising(
            recording=recording,
            parents=[self.waveform_extraction_node],
            model_folder_path=self.pca_model_folder_path,
        )

        num_iterations = 3
        tresholds = [5.0, 3.0, 1.0]
        peak_detector = IterativePeakDetector(
            recording=recording,
            peak_detector_node=peak_detector_node,
            waveform_extraction_node=self.waveform_extraction_node,
            waveform_denoising_node=waveform_denoising_node,
            num_iterations=num_iterations,
            return_output=(True, True),
            tresholds=tresholds,
        )

        self.peak_detector = peak_detector

        ##################
        # Extract features
        ##################

    def sort(self, job_kwargs=None):
        # Update job kwargs if needed
        job_kwargs = job_kwargs if job_kwargs is not None else self.job_kwargs
        nodes = [self.peak_detector]
        peaks, iterative_waveforms = run_node_pipeline(recording=self.recording, nodes=nodes, job_kwargs=job_kwargs)

        from spikeinterface.sortingcomponents.peak_pipeline import PeakRetriever

        peak_retriever = PeakRetriever(recording=self.recording, peaks=peaks)

        ms_before = 1.0
        ms_after = 1.0
        waveform_extraction_node = ExtractDenseWaveforms(
            recording=recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after
        )
        pca_projection_node = TemporalPCAProjection(
            recording=self.recording,
            model_folder_path=self.pca_model_folder_path,
            parents=[peak_retriever, waveform_extraction_node],
        )

        projections = run_node_pipeline(
            recording=self.recording,
            nodes=[peak_retriever, waveform_extraction_node, pca_projection_node],
            job_kwargs=job_kwargs,
        )

        print(projections.shape)


if __name__ == "__main__":
    from spikeinterface.core.datasets import download_dataset

    local_folder = download_dataset()
    from spikeinterface.extractors.neoextractors.mearec import MEArecRecordingExtractor, MEArecSortingExtractor

    recording = MEArecRecordingExtractor(local_folder)
    sorting = MEArecSortingExtractor(local_folder)

    job_kwargs = dict(n_jobs=1, chunk_size=30_000)
    sorter = ColumbiaSorter2022(recording, job_kwargs=job_kwargs)
    sorter.sort()
