"""
kwik structure based on:
https://github.com/kwikteam/phy-doc/blob/master/docs/kwik-format.md

cluster_group defaults based on:
https://github.com/kwikteam/phy-doc/blob/master/docs/kwik-model.md

04/08/20
"""

from __future__ import annotations


from pathlib import Path

import numpy as np

from spikeinterface.core import BaseRecording, BaseSorting, BaseRecordingSegment, BaseSortingSegment, read_python
from spikeinterface.core.core_tools import define_function_from_class


# noinspection SpellCheckingInspection
class KlustaSortingExtractor(BaseSorting):
    """Load Klusta format data as a sorting extractor.

    Parameters
    ----------
    file_or_folder_path : str or Path
        Path to the ALF folder.
    exclude_cluster_groups : list or str, default: None
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).

    Returns
    -------
    extractor : KlustaSortingExtractor
        The loaded data.
    """

    installation_mesg = "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"

    default_cluster_groups = {0: "Noise", 1: "MUA", 2: "Good", 3: "Unsorted"}

    def __init__(self, file_or_folder_path, exclude_cluster_groups=None):
        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        kwik_file_or_folder = Path(file_or_folder_path)
        kwikfile = None
        klustafolder = None
        if kwik_file_or_folder.is_file():
            assert kwik_file_or_folder.suffix == ".kwik", "Not a '.kwik' file"
            kwikfile = Path(kwik_file_or_folder).absolute()
            klustafolder = kwikfile.parent
        elif kwik_file_or_folder.is_dir():
            klustafolder = kwik_file_or_folder
            kwikfiles = [f for f in kwik_file_or_folder.iterdir() if f.suffix == ".kwik"]
            if len(kwikfiles) == 1:
                kwikfile = kwikfiles[0]
        assert kwikfile is not None, "Could not load '.kwik' file"

        try:
            config_file = [f for f in klustafolder.iterdir() if f.suffix == ".prm"][0]
            config = read_python(str(config_file))
            sampling_frequency = config["traces"]["sample_rate"]
        except Exception as e:
            print("Could not load sampling frequency info")

        kf_reader = h5py.File(kwikfile, "r")
        spiketrains = []
        unit_ids = []
        unique_units = []
        klusta_units = []
        cluster_groups_name = []
        groups = []
        unit = 0

        cs_to_exclude = []
        valid_group_names = [i[1].lower() for i in self.default_cluster_groups.items()]
        if exclude_cluster_groups is not None:
            assert isinstance(exclude_cluster_groups, list), "exclude_cluster_groups should be a list"
            for ec in exclude_cluster_groups:
                assert ec in valid_group_names, f"select exclude names out of: {valid_group_names}"
                cs_to_exclude.append(ec.lower())

        for channel_group in kf_reader.get("/channel_groups"):
            chan_cluster_id_arr = kf_reader.get(f"/channel_groups/{channel_group}/spikes/clusters/main")[()]
            chan_cluster_times_arr = kf_reader.get(f"/channel_groups/{channel_group}/spikes/time_samples")[()]
            chan_cluster_ids = np.unique(chan_cluster_id_arr)  # if clusters were merged in gui,
            # the original id's are still in the kwiktree, but
            # in this array

            for cluster_id in chan_cluster_ids:
                cluster_frame_idx = np.nonzero(chan_cluster_id_arr == cluster_id)  # the [()] is a h5py thing
                st = chan_cluster_times_arr[cluster_frame_idx]
                assert st.shape[0] > 0, "no spikes in cluster"
                cluster_group = kf_reader.get(f"/channel_groups/{channel_group}/clusters/main/{cluster_id}").attrs[
                    "cluster_group"
                ]

                assert (
                    cluster_group in self.default_cluster_groups.keys()
                ), f'cluster_group not in "default_dict: {cluster_group}'
                cluster_group_name = self.default_cluster_groups[cluster_group]

                if cluster_group_name.lower() in cs_to_exclude:
                    continue

                spiketrains.append(st)

                klusta_units.append(int(cluster_id))
                unique_units.append(unit)
                unit += 1
                groups.append(int(channel_group))
                cluster_groups_name.append(cluster_group_name)

        if len(np.unique(klusta_units)) == len(np.unique(unique_units)):
            unit_ids = klusta_units
        else:
            print("Klusta units are not unique! Using unique unit ids")
            unit_ids = unique_units

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.extra_requirements.append("h5py")

        self.add_sorting_segment(KlustSortingSegment(unit_ids, spiketrains))

        self.set_property("group", groups)
        quality = [e.lower() for e in cluster_groups_name]
        self.set_property("quality", quality)

        self._kwargs = {
            "file_or_folder_path": str(Path(file_or_folder_path).absolute()),
            "exclude_cluster_groups": exclude_cluster_groups,
        }


class KlustSortingSegment(BaseSortingSegment):
    def __init__(self, unit_ids, spiketrains):
        BaseSortingSegment.__init__(self)
        self._unit_ids = list(unit_ids)
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        unit_index = self._unit_ids.index(unit_id)
        times = self._spiketrains[unit_index]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times


read_klusta = define_function_from_class(source_class=KlustaSortingExtractor, name="read_klusta")
