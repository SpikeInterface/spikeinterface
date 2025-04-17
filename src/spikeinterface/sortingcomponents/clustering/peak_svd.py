from pathlib import Path
import pickle
import json

import numpy as np

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import (
    TemporalPCAProjection,
    MotionAwareTemporalPCAProjection,
)
from spikeinterface.core.node_pipeline import run_node_pipeline, ExtractSparseWaveforms, PeakRetriever


def extract_peaks_svd(
    recording,
    peaks,
    ms_before=0.5,
    ms_after=1.5,
    n_peaks_fit=5000,
    svd_model=None,
    n_components=5,
    radius_um=120.0,
    sparsity_mask=None,
    motion_aware=False,
    motion=None,
    folder=None,
    ensure_peak_same_sign=True,
    **job_kwargs,
):
    """
    Extract the sparse waveform compress to SVD (PCA) on a local set of channel per peak.
    So importantly, the output buffer have unaligned channel on shape[2].

    This is done in 2 steps:
      * fit a TruncatedSVD model on a few peaks on max channel
      * tranform each peaks in parralel on a sparse channel set with this model

    The recording have a drift, hen, optionally, the motion object can be given.
    In that case all the svd features are moved back using cubi interpolation.
    This avoid the use of interpolating the traces iself (with krigging).

    The output shape is (num_peaks, n_components, max_sparse_channel)
    """

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    # Step 1 : select a few peaks to fit the SVD
    if svd_model is None:
        few_peaks = select_peaks(
            peaks, recording=recording, method="uniform", n_peaks=n_peaks_fit, margin=(nbefore, nafter)
        )
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, job_name="Fit peaks svd", **job_kwargs
        )

        wfs = few_wfs[:, :, 0]
        from sklearn.decomposition import TruncatedSVD

        # Remove outliers
        valid = np.argmax(np.abs(wfs), axis=1) == nbefore
        wfs = wfs[valid]

        # Ensure all waveforms have a positive max
        if ensure_peak_same_sign:
            wfs *= -np.sign(wfs[:, nbefore])[:, np.newaxis]

        svd_model = TruncatedSVD(n_components=n_components)
        svd_model.fit(wfs)
        need_save_model = True
    else:
        need_save_model = False

    if folder is None:
        gather_mode = "memory"
        features_folder = None
        gather_kwargs = dict()
    else:
        gather_mode = "npy"
        if folder is None:
            raise ValueError("For gather_mode=npy a folder must be given")

        folder = Path(folder)

        # save the model
        if need_save_model:
            model_folder = folder / "svd_model"
            model_folder.mkdir(exist_ok=True, parents=True)
            with open(model_folder / "pca_model.pkl", "wb") as f:
                pickle.dump(svd_model, f)
            model_params = {
                "ms_before": ms_before,
                "ms_after": ms_after,
                "sampling_frequency": float(recording.sampling_frequency),
            }
            with open(model_folder / "params.json", "w") as f:
                json.dump(model_params, f)

        # save the features
        features_folder = folder / "features"
        gather_kwargs = dict(exist_ok=True)

    node0 = PeakRetriever(recording, peaks)

    if motion_aware:
        assert sparsity_mask is None
        # we need to increase the radius by the max motion
        max_motion = max(abs(e) for e in motion.get_boundaries())
        radius_um = radius_um + max_motion

    node1 = ExtractSparseWaveforms(
        recording,
        parents=[node0],
        return_output=False,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
        sparsity_mask=sparsity_mask,
    )

    if motion_aware:
        if motion is None:
            raise ValueError("For motion aware PCA motion must provided")
        node2 = MotionAwareTemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, pca_model=svd_model, motion=motion
        )
    else:
        node2 = TemporalPCAProjection(
            recording,
            parents=[node0, node1],
            return_output=True,
            pca_model=svd_model,
        )

    pipeline_nodes = [node0, node1, node2]

    peaks_svd = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        gather_mode=gather_mode,
        gather_kwargs=gather_kwargs,
        folder=features_folder,
        names=["sparse_svd"],
        job_name="Transform peaks svd",
    )

    sparse_mask = node1.neighbours_mask

    return peaks_svd, sparse_mask, svd_model
