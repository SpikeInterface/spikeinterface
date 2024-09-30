import numpy as np
from spikeinterface.core import Templates
from spikeinterface.core.node_pipeline import PeakDetector

_base_matching_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("cluster_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]

class BaseTemplateMatching(PeakDetector):
    def __init__(self, recording, templates, return_output=True, parents=None):
        # TODO make a sharedmem of template here
        # TODO maybe check that channel_id are the same with recording

        assert isinstance(templates, Templates), (
            f"The templates supplied is of type {type(templates)} and must be a Templates"
        )
        self.templates = templates
        PeakDetector.__init__(self, recording, return_output=return_output, parents=parents)

    def get_dtype(self):
        return np.dtype(_base_matching_dtype)

    def get_trace_margin(self):
        raise NotImplementedError  

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        spikes = self.compute_matching(traces, start_frame, end_frame, segment_index)
        spikes["segment_index"] = segment_index

        margin = self.get_trace_margin()
        if margin > 0:
            keep = (spikes["sample_index"] >= margin) & (spikes["sample_index"] < (traces.shape[0] - margin))
            spikes = spikes[keep]

        return spikes

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        raise NotImplementedError
    
    def get_extra_outputs(self):
        # can be overwritten if need to ouput some variables with a dict
        return None