from __future__ import annotations

from .external.combinato import CombinatoSorter
from .external.hdsort import HDSortSorter
from .external.herdingspikes import HerdingspikesSorter
from .external.ironclust import IronClustSorter
from .external.kilosort import KilosortSorter
from .external.kilosort2 import Kilosort2Sorter
from .external.kilosort2_5 import Kilosort2_5Sorter
from .external.kilosort3 import Kilosort3Sorter
from .external.kilosort4 import Kilosort4Sorter
from .external.pykilosort import PyKilosortSorter
from .external.klusta import KlustaSorter
from .external.mountainsort4 import Mountainsort4Sorter
from .external.mountainsort5 import Mountainsort5Sorter
from .external.spyking_circus import SpykingcircusSorter
from .external.tridesclous import TridesclousSorter
from .external.waveclus import WaveClusSorter
from .external.waveclus_snippets import WaveClusSnippetsSorter
from .external.yass import YassSorter

# based on spikeinertface.sortingcomponents
from .internal.spyking_circus2 import Spykingcircus2Sorter
from .internal.tridesclous2 import Tridesclous2Sorter
from .internal.simplesorter import SimpleSorter

sorter_full_list = [
    # external
    CombinatoSorter,
    HDSortSorter,
    HerdingspikesSorter,
    IronClustSorter,
    KilosortSorter,
    Kilosort2Sorter,
    Kilosort2_5Sorter,
    Kilosort3Sorter,
    Kilosort4Sorter,
    PyKilosortSorter,
    KlustaSorter,
    Mountainsort4Sorter,
    Mountainsort5Sorter,
    SpykingcircusSorter,
    TridesclousSorter,
    WaveClusSorter,
    WaveClusSnippetsSorter,
    YassSorter,
    # internal
    Spykingcircus2Sorter,
    Tridesclous2Sorter,
    SimpleSorter,
]

sorter_dict = {s.sorter_name: s for s in sorter_full_list}


def available_sorters():
    """Lists available sorters."""

    return sorted(list(sorter_dict.keys()))


def installed_sorters():
    """Lists installed sorters."""

    return sorted([s.sorter_name for s in sorter_full_list if s.is_installed()])


def print_sorter_versions():
    """ "Prints the versions of the installed sorters."""

    txt = ""
    for name in installed_sorters():
        version = sorter_dict[name].get_sorter_version()
        txt += "{}: {}\n".format(name, version)
    txt = txt[:-1]
    print(txt)


def get_default_sorter_params(sorter_name_or_class):
    """Returns default parameters for the specified sorter.

    Parameters
    ----------
    sorter_name_or_class : str or SorterClass
        The sorter to retrieve default parameters from.

    Returns
    -------
    default_params : dict
        Dictionary with default params for the specified sorter.
    """

    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError(f"Unknown sorter {sorter_name_or_class} has been given"))

    return SorterClass.default_params()


def get_sorter_params_description(sorter_name_or_class):
    """Returns a description of the parameters for the specified sorter.

    Parameters
    ----------
    sorter_name_or_class : str or SorterClass
        The sorter to retrieve parameters description from.

    Returns
    -------
    params_description : dict
        Dictionary with parameter description
    """

    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError(f"Unknown sorter {sorter_name_or_class} has been given"))

    return SorterClass.params_description()


def get_sorter_description(sorter_name_or_class):
    """Returns a brief description for the specified sorter.

    Parameters
    ----------
    sorter_name_or_class : str or SorterClass
        The sorter to retrieve description from.

    Returns
    -------
    params_description : dict
        Dictionary with parameter description.
    """

    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError(f"Unknown sorter {sorter_name_or_class} has been given"))

    return SorterClass.sorter_description
