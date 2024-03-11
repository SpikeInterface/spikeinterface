from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class

try:
    import pandas as pd

    HAVE_PANDAS = True
except:
    HAVE_PANDAS = False


class ALFSortingExtractor(BaseSorting):
    """Load ALF format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the ALF folder.
    sampling_frequency : int, default: 30000
        The sampling frequency.

    Returns
    -------
    extractor : ALFSortingExtractor
        The loaded data.
    """

    extractor_name = "ALFSorting"
    installed = HAVE_PANDAS
    installation_mesg = "To use the ALF extractors, install pandas: \n\n pip install pandas\n\n"
    name = "alf"

    def __init__(self, folder_path, sampling_frequency=30000):
        assert self.installed, self.installation_mesg
        # check correct parent folder:
        self._folder_path = Path(folder_path)
        if "probe" not in self._folder_path.name:
            raise ValueError('folder name should contain "probe", containing channels, clusters.* .npy datasets')
        # load datasets as mmap into a dict:
        required_alf_datasets = ["spikes.times", "spikes.clusters"]
        found_alf_datasets = dict()
        for alf_dataset_name in self.file_loc.iterdir():
            if "spikes" in alf_dataset_name.stem or "clusters" in alf_dataset_name.stem:
                if "npy" in alf_dataset_name.suffix:
                    dset = np.load(alf_dataset_name, mmap_mode="r", allow_pickle=True)
                    found_alf_datasets.update({alf_dataset_name.stem: dset})
                elif "metrics" in alf_dataset_name.stem:
                    found_alf_datasets.update({alf_dataset_name.stem: pd.read_csv(alf_dataset_name)})

        # check existence of datasets:
        if not any([i in found_alf_datasets for i in required_alf_datasets]):
            raise Exception(f"could not find {required_alf_datasets} in folder")

        spike_clusters = found_alf_datasets["spikes.clusters"]
        spike_times = found_alf_datasets["spikes.times"]

        # load units properties:
        total_units = 0
        properties = dict()

        for alf_dataset_name, alf_dataset in found_alf_datasets.items():
            if "clusters" in alf_dataset_name:
                if "clusters.metrics" in alf_dataset_name:
                    for property_name, property_values in found_alf_datasets[alf_dataset_name].iteritems():
                        properties[property_name] = property_values.tolist()
                else:
                    property_name = alf_dataset_name.split(".")[1]
                    properties[property_name] = alf_dataset
                    if total_units == 0:
                        total_units = alf_dataset.shape[0]

        if (
            "clusters.metrics" in found_alf_datasets
            and found_alf_datasets["clusters.metrics"].get("cluster_id") is not None
        ):
            unit_ids = found_alf_datasets["clusters.metrics"].get("cluster_id").tolist()
        else:
            unit_ids = list(range(total_units))

        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = ALFSortingSegment(spike_clusters, spike_times, sampling_frequency)
        self.add_sorting_segment(sorting_segment)

        self.extra_requirements.append("pandas")

        # add properties
        for property_name, values in properties.items():
            self.set_property(property_name, values)

        self._kwargs = {"folder_path": str(Path(folder_path).absolute()), "sampling_frequency": sampling_frequency}

    # @staticmethod
    # def write_sorting(sorting, save_path):
    #     assert HAVE_PANDAS, ALFSortingExtractor.installation_mesg
    #     # write cluster properties as clusters.<property_name>.npy
    #     save_path = Path(save_path)
    #     csv_property_names = ['cluster_id', 'cluster_id.1', 'num_spikes', 'firing_rate',
    #                           'presence_ratio', 'presence_ratio_std', 'frac_isi_viol',
    #                           'contamination_est', 'contamination_est2', 'missed_spikes_est',
    #                           'cum_amp_drift', 'max_amp_drift', 'cum_depth_drift', 'max_depth_drift',
    #                           'ks2_contamination_pct', 'ks2_label', 'amplitude_cutoff', 'amplitude_std',
    #                           'epoch_name', 'isi_viol']
    #     clusters_metrics_df = pd.DataFrame()
    #     for property_name in sorting.get_unit_property_names(0):
    #         data = sorting.get_units_property(property_name=property_name)
    #         if property_name not in csv_property_names:
    #             np.save(save_path / f'clusters.{property_name}', data)
    #         else:
    #             clusters_metrics_df[property_name] = data
    #     clusters_metrics_df.to_csv(save_path / 'clusters.metrics.csv')
    #     # save spikes.times, spikes.clusters
    #     clusters_number = []
    #     unit_spike_times = []
    #     for unit_no, unit_id in enumerate(sorting.get_unit_ids()):
    #         unit_spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
    #         if unit_spike_train is not None:
    #             unit_spike_times.extend(np.array(unit_spike_train) / sorting.get_sampling_frequency())
    #             clusters_number.extend([unit_no] * len(unit_spike_train))
    #     unit_spike_train = np.array(unit_spike_times)
    #     clusters_number = np.array(clusters_number)
    #     spike_times_ids = np.argsort(unit_spike_train)
    #     spike_times = unit_spike_train[spike_times_ids]
    #     spike_clusters = clusters_number[spike_times_ids]
    #     np.save(save_path / 'spikes.times', spike_times)
    #     np.save(save_path / 'spikes.clusters', spike_clusters)


class ALFSortingSegment(BaseSortingSegment):
    def __init__(self, spike_clusters, spike_times, sampling_frequency):
        self._spike_clusters = spike_clusters
        self._spike_times = spike_times
        self._sampling_frequency = sampling_frequency
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame,
        end_frame,
    ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf

        spike_times = self._spike_time[np.where(self._spike_clusters == unit_id)]
        spike_frames = spike_times * self._sampling_frequency
        return spike_frames[(spike_frames >= start_frame) & (spike_frames < end_frame)].astype("int64", copy=False)


read_alf_sorting = define_function_from_class(source_class=ALFSortingExtractor, name="read_alf_sorting")
