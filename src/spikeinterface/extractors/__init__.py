from .extractor_classes import *

from .toy_example import toy_example as toy_example
from .bids import read_bids as read_bids


from .neuropixels_utils import get_neuropixels_channel_groups, get_neuropixels_sample_shifts

from .neoextractors import get_neo_num_blocks, get_neo_streams

from warnings import warn


# deprecation of class import idea from neuroconv
# this __getattr__ is only triggered if the normal lookup fails so import
# any of our functions is fine but if someone tries to import a class this raises
# the warning and then returns the "function" version which will look the same
# to the end-user
# to be removed after version 0.105.0
def __getattr__(extractor_name):
    # we need this trick to allow us to use import * for spikeinterface.full
    if extractor_name == "__all__":
        __all__ = []
        for imp in globals():
            # need to remove a bunch of builtins etc that shouldn't be part of all
            if imp[0] != "_" and imp != "warn" and imp != "extractor_name":
                __all__.append(imp)
        return __all__
    all_extractors = list(recording_extractor_full_dict.values())
    all_extractors += list(sorting_extractor_full_dict.values())
    all_extractors += list(event_extractor_full_dict.values())
    all_extractors += list(snippets_extractor_full_dict.values())
    # special cases because they don't have simple wrappers
    # instead a single wrapper maps to multiple classes so we return
    # each class to check it
    from .neoextractors import (
        MEArecRecordingExtractor,
        MEArecSortingExtractor,
        OpenEphysBinaryEventExtractor,
        OpenEphysBinaryRecordingExtractor,
        OpenEphysLegacyRecordingExtractor,
        SpikeGLXEventExtractor,
    )

    all_extractors += [
        MEArecRecordingExtractor,
        MEArecSortingExtractor,
        OpenEphysBinaryEventExtractor,
        OpenEphysBinaryRecordingExtractor,
        OpenEphysLegacyRecordingExtractor,
        SpikeGLXEventExtractor,
    ]
    for reading_function in all_extractors:
        if extractor_name == reading_function.__name__:
            dep_msg = (
                "Importing classes at __init__ has been deprecated in favor of only importing function-size wrappers "
                "and will be removed in 0.105.0. For developers that prefer working with the class versions of extractors "
                "they can be imported from spikeinterface.extractors.extractor_classes"
                f"For class {reading_function.__name__}"
            )
            warn(dep_msg)
            return reading_function
    # this is necessary for objects that we don't support
    # normally this is an ImportError but since this is in the __getattr__ pytest needs an AttributeError
    raise AttributeError(f"cannot import name '{extractor_name}' from '{__name__}'")
