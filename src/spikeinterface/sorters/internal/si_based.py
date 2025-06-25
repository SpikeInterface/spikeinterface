from __future__ import annotations

from spikeinterface.core import load, NumpyRecording

from spikeinterface.sorters import BaseSorter


class ComponentsBasedSorter(BaseSorter):
    """
    This is a based class for sorter based on spikeinterface.sortingcomponents
    """

    @classmethod
    def is_installed(cls):
        return True

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        pass

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        sorting = load(output_folder / "sorting")
        return sorting

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False
