from pathlib import Path
import pickle
import json

import numpy as np

from spikeinterface.core import get_channel_distances, fix_job_kwargs
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
    seed=None,
    ensure_peak_same_sign=True,
    job_kwargs=None,
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

    job_kwargs = fix_job_kwargs(job_kwargs)

    nbefore = int(ms_before * recording.sampling_frequency / 1000.0)
    nafter = int(ms_after * recording.sampling_frequency / 1000.0)

    # Step 1 : select a few peaks to fit the SVD
    if svd_model is None:
        few_peaks = select_peaks(
            peaks, recording=recording, method="uniform", n_peaks=n_peaks_fit, margin=(nbefore, nafter), seed=seed
        )
        few_wfs = extract_waveform_at_max_channel(
            recording,
            few_peaks,
            ms_before=ms_before,
            ms_after=ms_after,
            job_name="Fit peaks svd",
            job_kwargs=job_kwargs,
        )

        wfs = few_wfs[:, :, 0]
        from sklearn.decomposition import TruncatedSVD

        # Remove outliers
        valid = np.argmax(np.abs(wfs), axis=1) == nbefore
        wfs = wfs[valid]

        # Ensure all waveforms have a positive max
        if ensure_peak_same_sign:
            wfs *= -np.sign(wfs[:, nbefore])[:, np.newaxis]

        svd_model = TruncatedSVD(n_components=n_components, random_state=seed)
        svd_model.fit(wfs)
    #     need_save_model = True
    # else:
    #     need_save_model = False

    if folder is None:
        gather_mode = "memory"
        features_folder = None
        gather_kwargs = dict()
    else:
        gather_mode = "npy"
        if folder is None:
            raise ValueError("For gather_mode=npy a folder must be given")

        folder = Path(folder)

        # # save the model
        # if need_save_model:
        #     model_folder = folder / "svd_model"
        #     model_folder.mkdir(exist_ok=True, parents=True)
        #     with open(model_folder / "pca_model.pkl", "wb") as f:
        #         pickle.dump(svd_model, f)
        #     model_params = {
        #         "ms_before": ms_before,
        #         "ms_after": ms_after,
        #         "sampling_frequency": float(recording.sampling_frequency),
        #     }
        #     with open(model_folder / "params.json", "w") as f:
        #         json.dump(model_params, f)

        # save the features
        features_folder = folder / "features"
        gather_kwargs = dict(exist_ok=True)

    node0 = PeakRetriever(recording, peaks)

    channel_distance = get_channel_distances(recording)

    if motion_aware:
        assert sparsity_mask is None
        # we need to increase the radius by the max motion for the waveforms mask
        # the final mask of svd will be th small one
        max_motion = max(abs(e) for e in motion.get_boundaries())
        margin = np.min(channel_distance[channel_distance > 0]) * 2
        # margin = 0
        wf_sparsity_mask = channel_distance <= (radius_um + max_motion + margin)
        final_sparsity_mask = channel_distance <= radius_um

    else:
        if sparsity_mask is None:
            wf_sparsity_mask = channel_distance <= radius_um
        else:
            wf_sparsity_mask = sparsity_mask

    node1 = ExtractSparseWaveforms(
        recording,
        parents=[node0],
        return_output=False,
        ms_before=ms_before,
        ms_after=ms_after,
        radius_um=radius_um,
        sparsity_mask=wf_sparsity_mask,
    )

    if motion_aware:
        if motion is None:
            raise ValueError("For motion aware PCA motion must provided")
        node2 = MotionAwareTemporalPCAProjection(
            recording,
            parents=[node0, node1],
            return_output=True,
            pca_model=svd_model,
            motion=motion,
            # interpolation_method="linear",
            interpolation_method="cubic",
            final_sparsity_mask=final_sparsity_mask,
        )
        out_names = ["sparse_svd", "peak_channel_index"]
    else:
        node2 = TemporalPCAProjection(
            recording,
            parents=[node0, node1],
            return_output=True,
            pca_model=svd_model,
        )
        out_names = ["sparse_svd"]

    pipeline_nodes = [node0, node1, node2]

    outs = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        gather_mode=gather_mode,
        gather_kwargs=gather_kwargs,
        folder=features_folder,
        names=out_names,
        job_name="Transform peaks svd",
    )

    if motion_aware:
        peaks_svd, peak_channel_indices = outs
        new_peaks = peaks.copy()
        new_peaks["channel_index"] = peak_channel_indices
        # here the mask is not the waveform mask (bigger) but the final mask of requested radius
        sparse_mask = final_sparsity_mask
        return peaks_svd, sparse_mask, svd_model, new_peaks
    else:
        peaks_svd = outs
        sparse_mask = wf_sparsity_mask
        return peaks_svd, sparse_mask, svd_model
