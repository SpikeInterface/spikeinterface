import copy
from typing import Dict
from spikeinterface import load_extractor
import numpy as np

from spikeinterface.core.core_tools import define_function_from_class

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.channelsaggregationrecording import ChannelsAggregationRecording, aggregate_channels


def group_preprocessing_by_property(recording_):  # TODO: rename
    """ """
    loop_recording = copy.deepcopy(recording_)
    preprocessing_step_list = []
    preprocessing_kwargs_list = []
    while True:
        # There is a general problem here in that the __init__ sets everything
        # up according to the full recording but then we split by channel.
        # However it is not possible to know what kwargs are user-passed and
        # what are setup during init.
        if isinstance(loop_recording, ChannelsAggregationRecording):
            raise ValueError(
                "Cannot group by recording a recording that" "has already been aggregated with `aggregate_channels`"
            )

        if (
            "recording" not in loop_recording._kwargs and "parent_recording" not in loop_recording._kwargs
        ):  # ZeroChannelPaddedRecording
            raise ValueError(f"The recording {loop_recording.__class__} " f"cannot be run in `group_by_recording`.")

        if "recording" in loop_recording._kwargs:  # ZeroChannelPaddedRecording
            assert "parent_recording" not in loop_recording._kwargs
            parent_recording = loop_recording._kwargs.pop("recording")
        else:
            parent_recording = loop_recording._kwargs.pop("parent_recording")

        if loop_recording.__class__ == spre.WhitenRecording:  # Hacky, discusss how to handle
            kwargs = copy.deepcopy(loop_recording._kwargs)
            kwargs["W"] = None
            kwargs["M"] = None
        else:
            kwargs = loop_recording._kwargs

        preprocessing_step_list.append(loop_recording.__class__)
        preprocessing_kwargs_list.append(kwargs)  # TODO: messy

        if isinstance(parent_recording, Dict):  # TODO: for silence periods, raise an issue.
            parent_recording = load_extractor(parent_recording)

        if "recording" not in parent_recording._kwargs:  # ZeroSilencePeriods
            root_recording = parent_recording
            break
        loop_recording = parent_recording

    split_recording = root_recording.split_by("group")

    preprocessed_recordings = []
    for shank_rec in split_recording.values():
        for pp_step, pp_kwargs in zip(preprocessing_step_list, preprocessing_kwargs_list):
            preprocessed_recordings.append(pp_step(shank_rec, **pp_kwargs))

    recombined_recording = aggregate_channels(preprocessed_recordings)

    return recombined_recording


from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import common_reference, highpass_spatial_filter
import spikeinterface.preprocessing as spre

num_channels = 84
recording = generate_recording(num_channels=num_channels)
groups = np.repeat([0, 1, 2, 3], num_channels / 4)
recording.set_property("group", groups)  # TODO: be clearler to copy

pp_recording = spre.whiten(recording)
pp_recording.get_traces(segment_index=0)

explicitly_per_shank = []
explicitly_split_recording = recording.split_by("group")

for shank_rec in explicitly_split_recording.values():
    explicitly_per_shank.append(spre.whiten(shank_rec))

agg_recording = aggregate_channels(explicitly_per_shank)
agg_recording.get_traces(segment_index=0)

recx = group_preprocessing_by_property(pp_recording)
recx.get_traces(segment_index=0)
# split_recording = group_preprocessing_by_property(pp_recording)

if False:
    from spikeinterface.core import generate_recording
    from spikeinterface.preprocessing import common_reference, highpass_spatial_filter

    num_channels = 84

    recording = generate_recording(num_channels=num_channels)
    groups = np.repeat([0, 1, 2, 3], num_channels / 4)
    recording.set_property("group", groups)  # TODO: be clearler to copy

    referenced_all_shanks = highpass_spatial_filter(recording, n_channel_pad=5)
    referenced_all_shanks_data = referenced_all_shanks.get_traces(segment_index=0)

    split_recording = group_by_recording(referenced_all_shanks)
    split_recording_data = split_recording.get_traces(segment_index=0)

    assert not np.array_equal(referenced_all_shanks_data, split_recording_data)

    # so certain 'all_traces' features are applied during

    # another sanity check (own test)
    explicitly_per_shank = []
    explicitly_split_recording = recording.split_by("group")

    for shank_rec in explicitly_split_recording.values():
        explicitly_per_shank.append(highpass_spatial_filter(shank_rec, n_channel_pad=5))

    explicitly_per_shank_rec = aggregate_channels(explicitly_per_shank)
    explicitly_per_shank_rec_data = explicitly_per_shank_rec.get_traces(segment_index=0)

    assert np.array_equal(split_recording_data, explicitly_per_shank_rec_data)  # TODO: why is this not exact?

    rec = aggregate_channels(explicitly_per_shank_rec)
    split_recording = group_by_recording(referenced_all_shanks)
    # check a second preprocessing step is not per-shank

    # also add 2 in a row to check this.
