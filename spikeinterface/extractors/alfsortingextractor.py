from spikeinterface.core import BinaryRecordingExtractor, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import write_binary_recording
from probeinterface import read_prb, write_prb

import json
import numpy as np
from pathlib import Path

try:
    import pandas as pd

    HAVE_PANDAS = True
except:
    HAVE_PANDAS = False


class ALFSortingExtractor(BaseSorting):
    extractor_name = 'ALFSorting'
    installed = HAVE_PANDAS
    is_writable = True
    installation_mesg = "To use the SHYBRID extractors, install SHYBRID: \n\n pip install shybrid\n\n"

    def __init__(self, folder_path, sampling_frequency=30000):
        assert self.installed, self.installation_mesg
        # check correct parent folder:
        self._folder_path = Path(folder_path)
        if 'probe' not in self._folder_path.name:
            raise ValueError('folder name should contain "probe", containing channels, clusters.* .npy datasets')
        # load datasets as mmap into a dict:
        required_alf_datasets = ['spikes.times', 'spikes.clusters']
        found_alf_datasets = dict()
        for alf_dataset_name in self.file_loc.iterdir():
            if 'spikes' in alf_dataset_name.stem or 'clusters' in alf_dataset_name.stem:
                if 'npy' in alf_dataset_name.suffix:
                    dset = np.load(alf_dataset_name, mmap_mode='r', allow_pickle=True)
                    found_alf_datasets.update({alf_dataset_name.stem: dset})
                elif 'metrics' in alf_dataset_name.stem:
                    found_alf_datasets.update({alf_dataset_name.stem: pd.read_csv(alf_dataset_name)})

        # check existence of datasets:
        if not any([i in found_alf_datasets for i in required_alf_datasets]):
            raise Exception(f'could not find {required_alf_datasets} in folder')

        # load units properties:
        total_units = 0
        properties = dict()

        for alf_dataset_name, alf_dataset in found_alf_datasets.items():
            if 'clusters' in alf_dataset_name:
                if 'clusters.metrics' in alf_dataset_name:
                    for property_name, property_values in found_alf_datasets[alf_dataset_name].iteritems():
                        properties[property_name] = property_values.tolist()
                else:
                    property_name = alf_dataset_name.split('.')[1]
                    properties[property_name] = alf_dataset
                    if total_units == 0:
                        total_units = alf_dataset.shape[0]

        if 'clusters.metrics' in found_alf_datasets and \
                found_alf_datasets['clusters.metrics'].get('cluster_id') is not None:
            unit_ids = found_alf_datasets['clusters.metrics'].get('cluster_id').tolist()
        else:
            unit_ids = list(range(total_units))

        units_map = {i: j for i, j in zip(self.get_unit_ids(), list(range(total_units)))}
        units_raster = []

        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'sampling_frequency': sampling_frequency}

    @staticmethod
    def write_sorting(sorting, save_path):
        """
        This is an example of a function that is not abstract so it is optional if you want to override it. It allows other
        SortingExtractors to use your new SortingExtractor to convert their sorted data into your
        sorting file format.
        """
        assert HAVE_PANDAS, ALFSortingExtractor.installation_mesg
        # write cluster properties as clusters.<property_name>.npy
        save_path = Path(save_path)
        csv_property_names = ['cluster_id', 'cluster_id.1', 'num_spikes', 'firing_rate',
                              'presence_ratio', 'presence_ratio_std', 'frac_isi_viol',
                              'contamination_est', 'contamination_est2', 'missed_spikes_est',
                              'cum_amp_drift', 'max_amp_drift', 'cum_depth_drift', 'max_depth_drift',
                              'ks2_contamination_pct', 'ks2_label', 'amplitude_cutoff', 'amplitude_std',
                              'epoch_name', 'isi_viol']
        clusters_metrics_df = pd.DataFrame()
        for property_name in sorting.get_unit_property_names(0):
            data = sorting.get_units_property(property_name=property_name)
            if property_name not in csv_property_names:
                np.save(save_path / f'clusters.{property_name}', data)
            else:
                clusters_metrics_df[property_name] = data
        clusters_metrics_df.to_csv(save_path / 'clusters.metrics.csv')
        # save spikes.times, spikes.clusters
        clusters_number = []
        unit_spike_times = []
        for unit_no, unit_id in enumerate(sorting.get_unit_ids()):
            unit_spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
            if unit_spike_train is not None:
                unit_spike_times.extend(np.array(unit_spike_train) / sorting.get_sampling_frequency())
                clusters_number.extend([unit_no] * len(unit_spike_train))
        unit_spike_train = np.array(unit_spike_times)
        clusters_number = np.array(clusters_number)
        spike_times_ids = np.argsort(unit_spike_train)
        spike_times = unit_spike_train[spike_times_ids]
        spike_clusters = clusters_number[spike_times_ids]
        np.save(save_path / 'spikes.times', spike_times)
        np.save(save_path / 'spikes.clusters', spike_clusters)


class ALFSortingSegment(BaseSortingSegment):
    def __init__(self, units_map):
        self._units_map = units_map
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame,
                             end_frame,
                             ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf

        cluster_sp_times = self._get_clusters_spike_times(unit_idx)
        if cluster_sp_times is None:
            return np.array([])
        max_frame = np.ceil(cluster_sp_times[-1] * self.get_sampling_frequency()).astype('int64')
        min_frame = np.floor(cluster_sp_times[0] * self.get_sampling_frequency()).astype('int64')
        start_frame = min_frame if start_frame is None or start_frame < min_frame else start_frame
        end_frame = max_frame if end_frame is None or end_frame > max_frame else end_frame
        if start_frame > max_frame or end_frame < min_frame:
            raise ValueError(f'Use start_frame to end_frame between {min_frame} and {max_frame}')
        cluster_sp_frames = (cluster_sp_times * self.get_sampling_frequency()).astype('int64')
        frame_idx = np.where((cluster_sp_frames >= start_frame) &
                             (cluster_sp_frames < end_frame))
        return cluster_sp_frames[frame_idx]

    def _get_clusters_spike_times(self, cluster_idx):
        if len(self._units_raster) == 0:
            spike_cluster_data = self._found_alf_datasets['spikes.clusters']
            spike_times_data = self._found_alf_datasets['spikes.times']
            df = pd.DataFrame({'sp_cluster': spike_cluster_data, 'sp_times': spike_times_data})
            data = df.groupby(['sp_cluster'])['sp_times'].apply(np.array).reset_index(name='sp_times_group')
            self._max_time = 0
            self._units_raster = [None] * self._total_units
            for index, sp_times_list in data.values:
                self._units_raster[index] = sp_times_list
                max_time = max(sp_times_list)
                if max_time > self._max_time:
                    self._max_time = max_time
        return self._units_raster[cluster_idx]

