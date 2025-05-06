from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractSparseWaveforms,
    ExtractDenseWaveforms,
    PeakRetriever,
    PipelineNode,
)
from spikeinterface.sortingcomponents.waveforms.temporal_pca import (
    TemporalPCAProjection,
)
from spikeinterface.core.job_tools import fix_job_kwargs
import numpy as np
from scipy.spatial.distance import cdist


class FindNearestTemplate(PipelineNode):
    def __init__(
        self,
        recording,
        pca_model,
        sparsity_mask,
        templates,
        name="nn_templates",
        return_output=True,
        parents=None,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)
        templates_array = templates.get_dense_templates()
        n_templates = templates_array.shape[0]
        num_channels = recording.get_num_channels()
        self.svd_templates = np.zeros((n_templates, pca_model.n_components, num_channels), "float32")
        for i in range(n_templates):
            self.svd_templates[i] = pca_model.transform(templates_array[i].T).T
        self.sparsity_mask = sparsity_mask
        self._dtype = recording.get_dtype()
        self._kwargs.update(
            dict(
                sparsity_mask=self.sparsity_mask,
                svd_templates=self.svd_templates,
            )
        )

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        peak_labels = np.empty(len(peaks), dtype="int64")
        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.sparsity_mask[main_chan])
            local_svds = waveforms[idx][:, :, : len(chan_inds)]
            XA = local_svds.reshape(local_svds.shape[0], -1)
            XB = self.svd_templates[:, :, chan_inds].reshape(self.svd_templates.shape[0], -1)
            distances = cdist(XA, XB, metric="euclidean")
            peak_labels[idx] = np.argmin(distances, axis=1)
        return peak_labels


def assign_templates_to_peaks(
    recording, 
    peaks, 
    ms_before, 
    ms_after, 
    svd_model, 
    sparse_mask, 
    templates, 
    gather_mode="memory",
    **job_kwargs
) -> np.ndarray | tuple[np.ndarray, dict]:

    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = PeakRetriever(recording, peaks)

    if templates.are_templates_sparse():
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
            sparsity_mask=sparse_mask,
        )
    else:
        node1 = ExtractDenseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
        )

    node2 = TemporalPCAProjection(
        recording,
        parents=[node0, node1],
        return_output=False,
        pca_model=svd_model,
    )

    node3 = FindNearestTemplate(
        recording,
        parents=[node0, node2],
        return_output=True,
        pca_model=svd_model,
        templates=templates,
        sparsity_mask=sparse_mask,
    )

    pipeline_nodes = [node0, node1, node2, node3]

    peak_labels = run_node_pipeline(
        recording,
        pipeline_nodes,
        job_kwargs,
        job_name=f"assign labels",
        gather_mode=gather_mode,
        squeeze_output=True,
    )
    return peak_labels
