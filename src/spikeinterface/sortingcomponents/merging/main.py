from __future__ import annotations

from threadpoolctl import threadpool_limits
import numpy as np



def merge_spikes(
    recording, sorting, method="circus", method_kwargs={}, extra_outputs=False, verbose=False, **job_kwargs
):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    sorting: Sorting
        The NumpySorting object
    method: "circus"
        Which method to use for merging spikes
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_outputs: bool
        If True then method_kwargs is also returned

    Returns
    -------
    new_sorting: NumpySorting
        Sorting found after merging
    method_kwargs:
        Optionaly returns for debug purpose.

    """
    from .method_list import merging_methods

    assert method in merging_methods, f"The 'method' {method} is not valid. Use a method from {merging_methods}"

    method_class = merging_methods[method]
    method_kwargs = method_class.initialize_and_check_kwargs(recording, sorting, method_kwargs)
    new_sorting = method_class.main_function(recording, sorting, method_kwargs)

    if extra_outputs:
        return new_sorting, method_kwargs
    else:
        return new_sorting


# generic class for template engine
class BaseMergingEngine:
    default_params = {}

    @classmethod
    def initialize_and_check_kwargs(cls, recording, sorting, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def main_function(cls, recording, sorting, method_kwargs):
        # need to be implemented in subclass
        raise NotImplementedError