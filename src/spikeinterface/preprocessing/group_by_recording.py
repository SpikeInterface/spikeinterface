import copy
from typing import Dict
from spikeinterface import load_extractor
import numpy as np


from spikeinterface.core.channelsaggregationrecording import ChannelsAggregationRecording, aggregate_channels
import spikeinterface.preprocessing as spre


def group_preprocessing_by_property(recording_):
    """
    1) Recursively loop through recording ._kwargs storing applied
       preprocessing steps and _kwargs until the initial recording is found
    2) Re-apply all steps back onto the base recording.
    """
    loop_recording = copy.deepcopy(recording_)
    preprocessing_step_list = []
    preprocessing_kwargs_list = []
    while True:
        # Recursively loop through recordings on kwargs, storing the preprocessing
        # steps and the associated kwargs ----------------------------------------------

        if isinstance(loop_recording, ChannelsAggregationRecording):
            raise ValueError(
                "Cannot group by recording a recording that" "has already been aggregated with `aggregate_channels`"
            )

        if (  # TODO: Handle ZeroChannelPaddedRecording that saves to 'parent_recording rather than 'recording'
            "recording" not in loop_recording._kwargs and "parent_recording" not in loop_recording._kwargs
        ):
            raise ValueError(f"The recording {loop_recording.__class__} " f"cannot be run in `group_by_recording`.")

        if "recording" in loop_recording._kwargs:
            assert "parent_recording" not in loop_recording._kwargs
            parent_recording = loop_recording._kwargs.pop("recording")
        else:
            parent_recording = loop_recording._kwargs.pop("parent_recording")

        # TODO: Hacky workaround to handle WhitenRecording
        if loop_recording.__class__ == spre.WhitenRecording:
            kwargs = copy.deepcopy(loop_recording._kwargs)
            kwargs["W"] = None
            kwargs["M"] = None
        else:
            kwargs = loop_recording._kwargs

        preprocessing_step_list.append(loop_recording.__class__)
        preprocessing_kwargs_list.append(kwargs)

        # TODO: ZeroSilencePeriods saves as a dict
        if isinstance(parent_recording, Dict):
            parent_recording = load_extractor(parent_recording)

        if "recording" not in parent_recording._kwargs:
            root_recording = parent_recording
            break
        loop_recording = parent_recording

    # Re-apply the preprocessing steps to the base function ----------------------------

    split_recording = root_recording.split_by("group")

    preprocessed_recordings = []
    for shank_rec in split_recording.values():
        for pp_step, pp_kwargs in zip(preprocessing_step_list, preprocessing_kwargs_list):
            preprocessed_recordings.append(pp_step(shank_rec, **pp_kwargs))

    recombined_recording = aggregate_channels(preprocessed_recordings)

    return recombined_recording
