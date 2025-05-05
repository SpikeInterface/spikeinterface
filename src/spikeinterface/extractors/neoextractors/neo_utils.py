from __future__ import annotations

from .neobaseextractor import NeoBaseRecordingExtractor


def get_neo_streams(extractor_name, *args, **kwargs):
    """Returns the NEO streams (stream names and stream ids) associated to a dataset.
    For multi-stream datasets, the `stream_id` or `stream_name` arguments can be used
    to select which stream to read with the `read_**extractor_name**()` function.

    Parameters
    ----------
    extractor_name : str
        The extractor name (available through the se.recording_extractor_full_dict).
    *args, **kwargs : arguments
        Extractor specific arguments. You can check extractor specific arguments with:
        `read_**extractor_name**?`


    Returns
    -------
    list
        List of NEO stream names
    list
        List of NEO stream ids
    """
    neo_extractor = get_neo_extractor(extractor_name)
    return neo_extractor.get_streams(*args, **kwargs)


def get_neo_num_blocks(extractor_name, *args, **kwargs) -> int:
    """Returns the number of NEO blocks.
    For multi-block datasets, the `block_index` argument can be used to select
    which bloack to read with the `read_**extractor_name**()` function.


    Parameters
    ----------
    extractor_name : str
        The extractor name (available through the se.recording_extractor_full_dict).
    *args, **kwargs : arguments
        Extractor specific arguments. You can check extractor specific arguments with:
        `read_**extractor_name**?`

    Returns
    -------
    int
        Number of NEO blocks

    Note
    ----
    Most datasets contain a single block.
    """
    neo_extractor = get_neo_extractor(extractor_name)
    return neo_extractor.get_num_blocks(*args, **kwargs)


def get_neo_extractor(extractor_name):
    from spikeinterface.extractors.extractorlist import recording_extractor_full_dict

    assert extractor_name in recording_extractor_full_dict, (
        f"{extractor_name} not an extractor name:" f"\n{list(recording_extractor_full_dict.keys())}"
    )
    neo_extractor = recording_extractor_full_dict[extractor_name]
    assert issubclass(neo_extractor, NeoBaseRecordingExtractor), f"{extractor_name} is not a NEO recording extractor!"
    return neo_extractor
