from .combinato import CombinatoSorter
from .hdsort import HDSortSorter
from .herdingspikes import HerdingspikesSorter

from .ironclust import IronClustSorter
from .kilosort import KilosortSorter
from .kilosort2 import Kilosort2Sorter
from .kilosort2_5 import Kilosort2_5Sorter
from .kilosort3 import Kilosort3Sorter
from .pykilosort import PyKilosortSorter
from .klusta import KlustaSorter
from .mountainsort4 import Mountainsort4Sorter
from .spyking_circus import SpykingcircusSorter
from .tridesclous import TridesclousSorter
from .waveclus import WaveClusSorter
from .yass import YassSorter

sorter_full_list = [
    CombinatoSorter,
    HDSortSorter,
    HerdingspikesSorter,
    IronClustSorter,
    KilosortSorter,
    Kilosort2Sorter,
    Kilosort2_5Sorter,
    Kilosort3Sorter,
    PyKilosortSorter,
    KlustaSorter,
    Mountainsort4Sorter,
    SpykingcircusSorter,
    TridesclousSorter,
    WaveClusSorter,
    YassSorter,
]

sorter_dict = {s.sorter_name: s for s in sorter_full_list}


def available_sorters():
    '''
    Lists available sorters.
    '''
    return sorted(list(sorter_dict.keys()))


def installed_sorters():
    '''
    Lists installed sorters.
    '''
    l = sorted([s.sorter_name for s in sorter_full_list if s.is_installed()])
    return l


def print_sorter_versions():
    txt = ''
    for name in installed_sorters():
        version = sorter_dict[name].get_sorter_version()
        txt += '{}: {}\n'.format(name, version)
    txt = txt[:-1]
    print(txt)


def get_default_params(sorter_name_or_class):
    '''
    Returns default parameters for the specified sorter.

    Parameters
    ----------
    sorter_name_or_class: str or SorterClass
        The sorter to retrieve default parameters from

    Returns
    -------
    default_params: dict
        Dictionary with default params for the specified sorter

    '''
    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError('Unknown sorter'))

    return SorterClass.default_params()


def get_params_description(sorter_name_or_class):
    '''
    Returns a description of the parameters for the specified sorter.

    Parameters
    ----------
    sorter_name_or_class: str or SorterClass
        The sorter to retrieve parameters description from

    Returns
    -------
    params_description: dict
        Dictionary with parameter description

    '''
    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError('Unknown sorter'))

    return SorterClass.params_description()


def get_sorter_description(sorter_name_or_class):
    '''
    Returns a brief description of the of the specified sorter.

    Parameters
    ----------
    sorter_name_or_class: str or SorterClass
        The sorter to retrieve description from

    Returns
    -------
    params_description: dict
        Dictionary with parameter description

    '''
    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise (ValueError('Unknown sorter'))

    return SorterClass.sorter_description
