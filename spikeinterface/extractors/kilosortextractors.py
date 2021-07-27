from pathlib import Path
import numpy as np

from .phyextractors import PhySortingExtractor


class KiloSortSortingExtractor(PhySortingExtractor):
    """
    SortingExtractor for a Kilosort output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    keep_good_only: bool
        If True, only Kilosort-labeled 'good' units are returned
    """
    extractor_name = 'KilosortSorting'
    installed = False  # check at class level if installed or not
    is_writable = False
    mode = 'folder'

    def __init__(self, folder_path, keep_good_only=False):
        PhySortingExtractor.__init__(self, folder_path)

        if keep_good_only:
            if 'KSLabel' in self.get_property_keys():
                kslabels = self.get_property("KSLabel")
                good_units = np.where(kslabels == "good")
                self._main_ids = self._main_ids[good_units]
            else:
                raise AttributeError("'KSLabel property not found in sorting properties!")

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'keep_good_only': keep_good_only}


def read_kilosort(*args, **kwargs):
    sorting = KiloSortSortingExtractor(*args, **kwargs)
    return sorting


read_kilosort.__doc__ = KiloSortSortingExtractor.__doc__
