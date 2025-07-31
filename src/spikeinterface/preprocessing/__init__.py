from .preprocessing_classes import *

from .motion import (
    correct_motion,
    load_motion_info,
    save_motion_info,
    get_motion_parameters_preset,
    get_motion_presets,
    compute_motion,
)

from .preprocessing_tools import get_spatial_interpolation_kernel
from .detect_bad_channels import detect_bad_channels
from .correct_lsb import correct_lsb

from .pipeline import (
    apply_preprocessing_pipeline,
    get_preprocessing_dict_from_analyzer,
    get_preprocessing_dict_from_file,
    PreprocessingPipeline,
)

# for snippets
from .align_snippets import AlignSnippets
from warnings import warn


# deprecation of class import idea from neuroconv
# this __getattr__ is only triggered if the normal lookup fails so import
# any of our functions is fine but if someone tries to import a class this raises
# the warning and then returns the "function" version which will look the same
# to the end-user
# to be removed after version 0.105.0
def __getattr__(preprocessor_name):
    # we need this trick to allow us to use import * for spikeinterface.full
    if preprocessor_name == "__all__":
        __all__ = []
        for imp in globals():
            # need to remove a bunch of builtins etc that shouldn't be part of all
            if imp[0] != "_" and imp != "warn" and imp != "preprocessor_name":
                __all__.append(imp)
        return __all__
    from .preprocessing_classes import _all_preprocesser_dict

    for pp_class, pp_function in _all_preprocesser_dict.items():
        if preprocessor_name == pp_class.__name__:
            dep_msg = (
                "Importing classes at __init__ has been deprecated in favor of only importing functions "
                "and will be removed in 0.105.0. For developers that prefer working with the class versions of preprocessors "
                "they can be imported from spikeinterface.preprocessors.preprocessor_classes."
            )
            warn(dep_msg)
            return pp_function
    # this is necessary for objects that we don't support
    raise AttributeError(f"cannot import name '{preprocessor_name}' from '{__name__}'")
