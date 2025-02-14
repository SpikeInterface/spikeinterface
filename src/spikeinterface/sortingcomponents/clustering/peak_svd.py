import numpy as np

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection, MotionAwareTemporalPCAProjection
from spikeinterface.core.node_pipeline import run_node_pipeline, ExtractSparseWaveforms, PeakRetriever



def extract_peaks_svd(
        recording,
        peaks,
        ms_before=0.5,
        ms_after=1.5,
        n_components=5,
        radius_um=120.,
        motion_aware=False,
        motion=None,
        **job_kwargs
    ):
    """
    Extract SVD feature from peaks from all channels or sparse channel.
    """

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    # Step 1 : select a few peaks to fit the SVD
    few_peaks = select_peaks(peaks, recording=recording, method="uniform", n_peaks=5000, margin=(nbefore, nafter))
    few_wfs = extract_waveform_at_max_channel(
        recording, few_peaks, ms_before=ms_before, ms_after=ms_after,
        job_name="extract some waveforms for fitting svd", **job_kwargs
    )

    wfs = few_wfs[:, :, 0]
    from sklearn.decomposition import TruncatedSVD

    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(wfs)

    # model_folder = clustering_folder / "tsvd_model"

    # model_folder.mkdir(exist_ok=True)
    # with open(model_folder / "pca_model.pkl", "wb") as f:
    #     pickle.dump(tsvd, f)

    # model_params = {
    #     "ms_before": ms_before,
    #     "ms_after": ms_after,
    #     "sampling_frequency": float(sampling_frequency),
    # }
    # with open(model_folder / "params.json", "w") as f:
    #     json.dump(model_params, f)

    # features

    # features_folder = clustering_folder / "features"
    node0 = PeakRetriever(recording, peaks)

    node1 = ExtractSparseWaveforms(
        recording,
        parents=[node0],
        return_output=False,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
    )

    # model_folder_path = clustering_folder / "tsvd_model"

    if motion_aware:
        if motion is None:
            raise ValueError("For motion aware PCA motion must provided")
        node2 = MotionAwareTemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, pca_model=tsvd, motion=motion
        )
    else:
        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, pca_model=tsvd,
        )

    pipeline_nodes = [node0, node1, node2]

    peaks_svd = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        # gather_mode="npy",
        gather_mode="memory",
        # gather_kwargs=dict(exist_ok=True),
        # folder=features_folder,
        names=["sparse_wfs", "sparse_tsvd"],
        job_name="Extract peaks svd",
    )
    
    sparse_mask = node1.neighbours_mask

    return peaks_svd, sparse_mask