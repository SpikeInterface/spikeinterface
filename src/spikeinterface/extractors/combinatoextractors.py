from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class CombinatoSortingExtractor(BaseSorting):
    """Load Combinato format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the Combinato folder.
    sampling_frequency : int, default: 30000
        The sampling frequency.
    user : str, default: "simple"
        The username that ran the sorting
    det_sign : "both", "pos", "neg", default: "both"
        Which sign was used for detection.
    keep_good_only : bool, default: True
        Whether to only keep good units.

    Returns
    -------
    extractor : CombinatoSortingExtractor
        The loaded data.
    """

    installation_mesg = "To use the CombinatoSortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, folder_path, sampling_frequency=None, user="simple", det_sign="both", keep_good_only=True):
        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        folder_path = Path(folder_path)
        assert folder_path.is_dir(), "Folder {} doesn't exist".format(folder_path)
        if sampling_frequency is None:
            h5_path = str(Path(folder_path).absolute()) + ".h5"
            if Path(h5_path).exists():
                with h5py.File(h5_path, mode="r") as f:
                    sampling_frequency = f["sr"][0]

        # ~ self.set_sampling_frequency(sampling_frequency)
        det_file = str(folder_path / Path("data_" + folder_path.stem + ".h5"))
        sort_cat_files = []
        for sign in ["neg", "pos"]:
            if det_sign in ["both", sign]:
                sort_cat_file = folder_path / Path("sort_{}_{}/sort_cat.h5".format(sign, user))
                if sort_cat_file.exists():
                    sort_cat_files.append((sign, str(sort_cat_file)))

        unit_counter = 0
        spiketrains = {}
        metadata = {}
        unsorted = []
        with h5py.File(det_file, mode="r") as fdet:
            for sign, sfile in sort_cat_files:
                with h5py.File(sfile, mode="r") as f:
                    sp_class = f["classes"][()]
                    gaux = f["groups"][()]
                    groups = {g: gaux[gaux[:, 1] == g, 0] for g in np.unique(gaux[:, 1])}  # array of classes per group
                    group_type = {group: g_type for group, g_type in f["types"][()]}
                    sp_index = f["index"][()]

                times_css = fdet[sign]["times"][()]
                for gr, cls in groups.items():
                    if keep_good_only and (group_type[gr] < 1):  # artifact or unsorted
                        continue
                    spiketrains[unit_counter] = np.rint(
                        times_css[sp_index[np.isin(sp_class, cls)]] * (sampling_frequency / 1000)
                    )
                    metadata[unit_counter] = {"group_type": group_type[gr]}
                    unit_counter = unit_counter + 1
        unit_ids = np.arange(unit_counter, dtype="int64")
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(CombinatoSortingSegment(spiketrains))
        self.set_property("unsorted", np.array([metadata[u]["group_type"] == 0 for u in range(unit_counter)]))
        self.set_property("artifact", np.array([metadata[u]["group_type"] == -1 for u in range(unit_counter)]))
        self._kwargs = {"folder_path": str(Path(folder_path).absolute()), "user": user, "det_sign": det_sign}

        self.extra_requirements.append("h5py")


class CombinatoSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains):
        BaseSortingSegment.__init__(self)
        # spiketrains is dict
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._spiketrains[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_combinato = define_function_from_class(source_class=CombinatoSortingExtractor, name="read_combinato")
