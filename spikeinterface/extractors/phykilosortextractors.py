from pathlib import Path
import numpy as np

from spikeinterface.core import (BaseSorting, BaseSortingSegment, read_python)


class BasePhyKilosortSortingExtractor(BaseSorting):
    """
    Base SortingExtractor for Phy and Kilosort output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    exclude_cluster_groups: list or str (optional)
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"])
    """
    extractor_name = 'BasePhyKilosortSorting'
    installed = False  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = "To use the PhySortingExtractor install pandas: \n\n pip install pandas\n\n"  # error message when not installed

    def __init__(self, folder_path, exclude_cluster_groups=None, keep_good_only=False):
        try:
            import pandas as pd
            HAVE_PD = True
        except ImportError:
            HAVE_PD = False
        assert HAVE_PD, self.installation_mesg

        phy_folder = Path(folder_path)

        spike_times = np.load(phy_folder / 'spike_times.npy')
        spike_templates = np.load(phy_folder / 'spike_templates.npy')

        if (phy_folder / 'spike_clusters.npy').is_file():
            spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            spike_clusters = spike_templates

        clust_id = np.unique(spike_clusters)
        unit_ids = list(clust_id)
        spike_times.astype(int)
        params = read_python(str(phy_folder / 'params.py'))
        sampling_frequency = params['sample_rate']

        # try to load cluster info
        cluster_info_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']
                              and "cluster_info" in p.name]
        if len(cluster_info_files) == 1:
            cluster_info_file = cluster_info_files[0]
            if cluster_info_file.suffix == ".tsv":
                delimeter = "\t"
            else:
                delimeter = ","
            cluster_info = pd.read_csv(cluster_info_file, delimiter=delimeter)

        else:
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
                    cluster_info = pd.merge(cluster_info, new_property, on='cluster_id')

            cluster_info["id"] = cluster_info["cluster_id"]
            del cluster_info["cluster_id"]

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(f"group != '{exclude_cluster_groups}'")
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query(f"KSLabel != 'good'")

        unit_ids = cluster_info["id"].values
        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        for prop_name in cluster_info.columns:
            if prop_name in ['chan_grp', 'ch_group']:
                self.set_property(key="group", values=cluster_info[prop_name])
            elif prop_name != "group":
                self.set_property(key=prop_name, values=cluster_info[prop_name])

        self.add_sorting_segment(PhySortingSegment(spike_times, spike_clusters))

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups}


class PhySortingSegment(BaseSortingSegment):
    def __init__(self, all_spikes, all_clusters):
        BaseSortingSegment.__init__(self)
        self._all_spikes = all_spikes
        self._all_clusters = all_clusters

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spike_times = self._all_spikes[self._all_clusters == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times.copy()


class PhySortingExtractor(BasePhyKilosortSortingExtractor):
    """
    Base SortingExtractor for Phy and Kilosort output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    exclude_cluster_groups: list or str (optional)
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"])
    """
    extractor_name = 'BasePhyKilosortSorting'

    def __init__(self, folder_path, exclude_cluster_groups=None):
        BasePhyKilosortSortingExtractor.__init__(self, folder_path, exclude_cluster_groups, keep_good_only=False)


class KiloSortSortingExtractor(BasePhyKilosortSortingExtractor):
    """
    SortingExtractor for a Kilosort output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    keep_good_only: bool
        If True, only Kilosort-labeled 'good' units are returned
    """
    extractor_name = 'KiloSortSorting'

    def __init__(self, folder_path, keep_good_only=False):
        BasePhyKilosortSortingExtractor.__init__(self, folder_path, exclude_cluster_groups=None,
                                                 keep_good_only=keep_good_only)


def read_phy(*args, **kwargs):
    sorting = PhySortingExtractor(*args, **kwargs)
    return sorting


read_phy.__doc__ = PhySortingExtractor.__doc__


def read_kilosort(*args, **kwargs):
    sorting = KiloSortSortingExtractor(*args, **kwargs)
    return sorting


read_kilosort.__doc__ = KiloSortSortingExtractor.__doc__
