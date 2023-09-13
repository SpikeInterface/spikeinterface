import copy

from spikeinterface.core.node_pipeline import PeakRetriever, run_node_pipeline


def run_peak_pipeline(
    recording,
    peaks,
    nodes,
    job_kwargs,
    gather_mode="memory",
    squeeze_output=True,
    folder=None,
    names=None,
):
    # TODO remove this soon
    import warnings

    warnings.warn("run_peak_pipeline() is deprecated use run_node_pipeline() instead", DeprecationWarning, stacklevel=2)

    node0 = PeakRetriever(recording, peaks)
    # because nodes are modified inplace (insert parent) they need to copy incase
    # the same pipeline is run several times
    nodes = copy.deepcopy(nodes)

    for node in nodes:
        if node.parents is None:
            node.parents = [node0]
        else:
            node.parents = [node0] + node.parents
    all_nodes = [node0] + nodes
    job_kwargs["job_name"] = "peak pipeline"
    outs = run_node_pipeline(
        recording,
        all_nodes,
        job_kwargs,
        gather_mode=gather_mode,
        squeeze_output=squeeze_output,
        folder=folder,
        names=names,
    )
    return outs
