from functools import partial

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class GenericPreprocessor(BasePreprocessor):
    def __init__(self, recording, function, **function_kwargs):
        super().__init__(recording)
        self._serializability["json"] = False

        # Heavy computation can be done at the __init__ if needed
        self.function_to_apply = partial(function, **function_kwargs)

        # Initialize segments
        for segment in recording._recording_segments:
            processed_segment = GenericPreprocessorSegment(segment, self.function_to_apply)
            self.add_recording_segment(processed_segment)

        self._kwargs = {"recording": recording, "func": function}
        self._kwargs.update(**function_kwargs)


class GenericPreprocessorSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, function_to_apply):
        super().__init__(parent_segment)
        self.function_to_apply = function_to_apply  # Function to apply to the traces

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Fetch the traces from the parent segment
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        # Apply the function to the traces
        return self.function_to_apply(traces)


generic_preprocessor = define_function_from_class(GenericPreprocessor, name="generic_preprocessor")
