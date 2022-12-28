from pathlib import Path

import numpy as np

from spikeinterface.core import (BaseSorting, BaseSortingSegment, read_python)
from spikeinterface.core.core_tools import define_function_from_class


class BasePhyKilosortSortingExtractor(BaseSorting):
    """Base SortingExtractor for Phy and Kilosort output folder.

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    exclude_cluster_groups: list or str, optional
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).
    keep_good_only : bool, optional, default: True
        Whether to only keep good units.
    """
    extractor_name = 'BasePhyKilosortSorting'
    installed = False  # check at class level if installed or not
    mode = 'folder'
    installation_mesg = "To use the PhySortingExtractor install pandas: \n\n pip install pandas\n\n"  # error message when not installed
    name = "phykilosort"

    def __init__(self, folder_path, exclude_cluster_groups=None, keep_good_only=False):
        try:
            import pandas as pd
            HAVE_PD = True
        except ImportError:
            HAVE_PD = False
        assert HAVE_PD, self.installation_mesg

        phy_folder = Path(folder_path)

        spike_times = np.load(phy_folder / 'spike_times.npy')

        if (phy_folder / 'spike_clusters.npy').is_file():
            spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            spike_clusters = np.load(phy_folder / 'spike_templates.npy')

        clust_id = np.unique(spike_clusters)
        unit_ids = list(clust_id)
        spike_times = spike_times.astype(int)
        params = read_python(str(phy_folder / 'params.py'))
        sampling_frequency = params['sample_rate']

        # try to load cluster info
        cluster_info_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']
                              and "cluster_info" in p.name]

        if len(cluster_info_files) == 1:
            # load properties from cluster_info file
            cluster_info_file = cluster_info_files[0]
            if cluster_info_file.suffix == ".tsv":
                delimeter = "\t"
            else:
                delimeter = ","
            cluster_info = pd.read_csv(cluster_info_file, delimiter=delimeter)
        else:
            # load properties from other tsv/csv files
            all_property_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']]

            cluster_info = None
            for file in all_property_files:
                if file.suffix == ".tsv":
                    delimeter = "\t"
                else:
                    delimeter = ","
                new_property = pd.read_csv(file, delimiter=delimeter)
                if cluster_info is None:
                    cluster_info = new_property
                else:
                    cluster_info = pd.merge(cluster_info, new_property, on='cluster_id', suffixes=[None, '_repeat'])

        # in case no tsv/csv files are found populate cluster info with minimal info
        if cluster_info is None:
            cluster_info = pd.DataFrame({'cluster_id': unit_ids})
            cluster_info['group'] = ['unsorted'] * len(unit_ids)

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(f"group != '{exclude_cluster_groups}'")
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        if "cluster_id" not in cluster_info.columns:
            assert "id" in cluster_info.columns, "Couldn't find cluster ids in the tsv files!"
            cluster_info.loc[:, "cluster_id"] = cluster_info["id"].values
            del cluster_info["id"]

        # update spike clusters and times values
        bad_clusters = [clust for clust in clust_id if clust not in cluster_info['cluster_id'].values]
        spike_clusters_clean_idxs = ~np.isin(spike_clusters, bad_clusters)
        spike_clusters_clean = spike_clusters[spike_clusters_clean_idxs]
        spike_times_clean = spike_times[spike_clusters_clean_idxs]

        if 'si_unit_id' in cluster_info.columns:
            unit_ids = cluster_info["si_unit_id"].values

            if np.all(np.isnan(unit_ids)):
                max_si_unit_id = -1
            else:
                max_si_unit_id = int(np.nanmax(unit_ids))

            for i, (phy_id, si_id) in enumerate(zip(cluster_info["cluster_id"].values,
                                                    cluster_info["si_unit_id"].values)):
                if np.isnan(si_id):
                    max_si_unit_id += 1
                    new_si_id = int(max_si_unit_id)
                else:
                    new_si_id = si_id
                unit_ids[i] = new_si_id

            # Little hack to replace values in spike_clusters_clean to spike_clusters_new very efficiently.
            from_values = cluster_info['cluster_id'].values
            sort_idx = np.argsort(from_values)
            idx = np.searchsorted(from_values, spike_clusters_clean, sorter=sort_idx)
            spike_clusters_new = unit_ids[sort_idx][idx]

            unit_ids = unit_ids.astype(int)
            spike_clusters_clean = spike_clusters_new
            del cluster_info["si_unit_id"]
        else:
            unit_ids = cluster_info["cluster_id"].values

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.extra_requirements.append('pandas')

        del cluster_info["cluster_id"]
        for prop_name in cluster_info.columns:
            if prop_name in ['chan_grp', 'ch_group']:
                self.set_property(key="group", values=cluster_info[prop_name])
            elif prop_name != "group":
                self.set_property(key=prop_name, values=cluster_info[prop_name])
            elif prop_name == "group":
                # rename group property to 'quality'
                self.set_property(key="quality", values=cluster_info[prop_name])

        self.add_sorting_segment(PhySortingSegment(spike_times_clean, spike_clusters_clean))


class PhySortingSegment(BaseSortingSegment):
    def __init__(self, all_spikes, all_clusters):
        BaseSortingSegment.__init__(self)
        self._all_spikes = all_spikes
        self._all_clusters = all_clusters

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        start = 0 if start_frame is None else np.searchsorted(self._all_spikes, start_frame, side="left")
        end = len(self._all_spikes) if end_frame is None else np.searchsorted(self._all_spikes, end_frame, side="right")

        spike_times = self._all_spikes[start:end][self._all_clusters[start:end] == unit_id]
        return np.atleast_1d(spike_times.copy().squeeze())


class PhySortingExtractor(BasePhyKilosortSortingExtractor):
    """Load Phy format data as a sorting extractor.

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py).
    exclude_cluster_groups: list or str, optional
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).

    Returns
    -------
    extractor : PhySortingExtractor
        The loaded data.
    """
    extractor_name = 'PhySorting'
    name = "phy"

    def __init__(self, folder_path, exclude_cluster_groups=None):
        BasePhyKilosortSortingExtractor.__init__(self, folder_path, exclude_cluster_groups, keep_good_only=False)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups}


class KiloSortSortingExtractor(BasePhyKilosortSortingExtractor):
    """Load Kilosort format data as a sorting extractor.

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py).
    exclude_cluster_groups: list or str, optional
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"]).
    keep_good_only : bool, optional, default: True
        Whether to only keep good units.
        If True, only Kilosort-labeled 'good' units are returned.

    Returns
    -------
    extractor : KiloSortSortingExtractor
        The loaded data.
    """
    extractor_name = 'KiloSortSorting'
    name = "kilosort"

    def __init__(self, folder_path, keep_good_only=False):
        BasePhyKilosortSortingExtractor.__init__(self, folder_path, exclude_cluster_groups=None,
                                                 keep_good_only=keep_good_only)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'keep_good_only': keep_good_only}


read_phy = define_function_from_class(source_class=PhySortingExtractor, name="read_phy")
read_kilosort = define_function_from_class(source_class=KiloSortSortingExtractor, name="read_kilosort")
