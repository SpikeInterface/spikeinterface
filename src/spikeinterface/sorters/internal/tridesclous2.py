from .si_based import ComponentsBasedSorter

from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels, extract_waveforms, NumpySorting, get_channel_distances
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore

import numpy as np

import pickle
import json

class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = "tridesclous2"

    _default_params = {
        "apply_preprocessing": True,
        "waveforms" : {"ms_before": 0.5, "ms_after": 1.5, },
        "filtering": {"freq_min": 300, "freq_max": 8000.0},
        "detection": {"peak_sign": "neg", "detect_threshold": 5, "exclude_sweep_ms": 0.4, "radius_um": 100},
        "hdbscan_kwargs": {
            "min_cluster_size": 25,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
        },
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 20000},
        "svd": {"n_components": 6},
        "clustering": {
            "split_radius_um": 40.,
            "merge_radius_um": 40.,
        },
        "templates": {
            "ms_before": 1.5,
            "ms_after": 2.5,
            # "peak_shift_ms": 0.2,
        },
        "matching": {
            "peak_shift_ms": 0.2,
            "radius_um": 100.
        },
        "job_kwargs": {},
        "save_array": True,
    }

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        # this is importanted only on demand because numba import are too heavy
        # from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        # from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        # from spikeinterface.sortingcomponents.peak_selection import select_peaks
        # from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

        from spikeinterface.sortingcomponents.peak_pipeline import run_node_pipeline, ExtractDenseWaveforms, ExtractSparseWaveforms, PeakRetriever
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks, DetectPeakLocallyExclusive
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.peak_localization import  LocalizeCenterOfMass, LocalizeGridConvolution
        from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection

        from spikeinterface.sortingcomponents.clustering.split import split_clusters
        from spikeinterface.sortingcomponents.clustering.merge import merge_clusters
        from spikeinterface.sortingcomponents.clustering.tools import compute_template_from_sparse

        from sklearn.decomposition import TruncatedSVD

        import hdbscan

        recording_raw = load_extractor(
            sorter_output_folder.parent / "spikeinterface_recording.json", base_folder=sorter_output_folder.parent
        )

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            recording = bandpass_filter(recording_raw, **params["filtering"])
            # TODO this about zscore>common_reference or the reverse
            recording = common_reference(recording)
            recording = zscore(recording, dtype="float32")
            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording = recording_raw
            noise_levels = get_noise_levels(recording, return_scaled=False)

        # detection
        detection_params = params["detection"].copy()
        # detection_params["radius_um"] = params["general"]["radius_um"]
        detection_params["noise_levels"] = noise_levels
        all_peaks = detect_peaks(recording, method="locally_exclusive", **detection_params, **job_kwargs)

        if verbose:
            print("We found %d peaks in total" % len(all_peaks))

        # selection
        selection_params = params["selection"].copy()
        # selection_params["n_peaks"] = params["selection"]["n_peaks_per_channel"] * num_chans
        # selection_params["n_peaks"] = max(selection_params["min_n_peaks"], selection_params["n_peaks"])
        # selection_params["noise_levels"] = noise_levels
        # some_peaks = select_peaks(
        #     peaks, method="smart_sampling_amplitudes", select_per_channel=False, **selection_params
        # )
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)



        if verbose:
            print("We kept %d peaks for clustering" % len(peaks))

        # localization
        # localization_params = params["localization"].copy()
        # localization_params["radius_um"] = params["general"]["radius_um"]
        # peak_locations = localize_peaks(
        #     recording, some_peaks, method="monopolar_triangulation", **localization_params, **job_kwargs
        # )

        # ~ print(peak_locations.dtype)

        # features = localisations only
        # peak_features = np.zeros((peak_locations.size, 3), dtype="float64")
        # for i, dim in enumerate(["x", "y", "z"]):
        #     peak_features[:, i] = peak_locations[dim]

        # clusering is hdbscan

        # out = hdbscan.hdbscan(peak_features, **params["hdbscan_kwargs"])
        # peak_labels = out[0]

        # mask = peak_labels >= 0
        # labels = np.unique(peak_labels[mask])

        # extract waveform for template matching
        # sorting_temp = NumpySorting.from_times_labels(
        #     some_peaks["sample_index"][mask], peak_labels[mask], sampling_frequency
        # )
        # sorting_temp = sorting_temp.save(folder=sorter_output_folder / "sorting_temp")
        # waveforms_params = params["waveforms"].copy()
        # waveforms_params["ms_before"] = params["general"]["ms_before"]
        # waveforms_params["ms_after"] = params["general"]["ms_after"]
        # we = extract_waveforms(
        #     recording, sorting_temp, sorter_output_folder / "waveforms_temp", **waveforms_params, **job_kwargs
        # )

        ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces
        # matching_params = params["matching"].copy()
        # matching_params["waveform_extractor"] = we
        # matching_params["noise_levels"] = noise_levels
        # matching_params["peak_sign"] = params["detection"]["peak_sign"]
        # matching_params["detect_threshold"] = params["detection"]["detect_threshold"]
        # matching_params["radius_um"] = params["general"]["radius_um"]

        # TODO: route that params
        # ~ 'num_closest' : 5,
        # ~ 'sample_shift': 3,
        # ~ 'ms_before': 0.8,
        # ~ 'ms_after': 1.2,
        # ~ 'num_peeler_loop':  2,
        # ~ 'num_template_try' : 1,

        # spikes = find_spikes_from_templates(
        #     recording, method="tridesclous", method_kwargs=matching_params, **job_kwargs
        # )

        # if verbose:
        #     print("We found %d spikes" % len(spikes))


        # SVD for time compression
        few_peaks = select_peaks(peaks, method="uniform", n_peaks=5000)
        few_wfs = extract_best_channel_waveform_chan(recording, few_peaks, **job_kwargs)
        wfs = few_wfs[:, :, 0]
        tsvd = TruncatedSVD(params["svd"]["n_components"])
        tsvd.fit(wfs)

        model_folder = sorter_output_folder  / 'tsvd_model'
        
        model_folder.mkdir(exist_ok=True)
        with open(model_folder / "pca_model.pkl", "wb") as f:
            pickle.dump(tsvd, f)

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]
        model_params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": float(sampling_frequency),
        }
        with open(model_folder / "params.json", "w") as f:
            json.dump(model_params, f)

        # features

        features_folder = sorter_output_folder  / 'features'
        node0 = PeakRetriever(recording, peaks)

        # node1 = ExtractDenseWaveforms(rec, parents=[node0], return_output=False,
        #     ms_before=0.5,
        #     ms_after=1.5, 
        # )

        # node2 = LocalizeCenterOfMass(rec, parents=[node0, node1], return_output=True,
        #                              local_radius_um=75.0,
        #                              feature="ptp", )

        # node2 = LocalizeGridConvolution(rec, parents=[node0, node1], return_output=True,
        #                             local_radius_um=40.,
        #                             upsampling_um=5.0,
        #                             )

        node3 = ExtractSparseWaveforms(recording, parents=[node0], return_output=True,
                                    ms_before=0.5,
                                    ms_after=1.5,
                                    radius_um=100.0,
        )

        model_folder_path = sorter_output_folder  / 'tsvd_model'

        node4 = TemporalPCAProjection(recording, parents=[node0, node3], return_output=True,
                                    model_folder_path=model_folder_path)


        # pipeline_nodes = [node0, node1, node2, node3, node4]
        pipeline_nodes = [node0, node3, node4]

        output = run_node_pipeline(recording, pipeline_nodes, job_kwargs, gather_mode="npy", gather_kwargs=dict(exist_ok=True),
                                folder=features_folder, names=["sparse_wfs", "sparse_tsvd"])

        # TODO make this generic in GatherNPY ???
        sparse_mask = node3.neighbours_mask
        np.save(features_folder/ 'sparse_mask.npy', sparse_mask)
        np.save(features_folder/ 'peaks.npy', peaks)
        


        # Clustering: channel index > split > merge
        split_radius_um = params["clustering"]["split_radius_um"]
        neighbours_mask = get_channel_distances(recording) < split_radius_um

        original_labels = peaks['channel_index']

        post_split_label, split_count = split_clusters(
            original_labels,
            recording,
            features_folder,
            method="hdbscan_on_local_pca",
            method_kwargs=dict(
                clusterer="hdbscan",
                # clusterer="isocut5",

                feature_name="sparse_tsvd",
                # feature_name="sparse_wfs",
                
                neighbours_mask=neighbours_mask,
                waveforms_sparse_mask=sparse_mask,
                min_size_split=50,
                min_cluster_size=50,
                min_samples=50,
                n_pca_features=3,
                ),

            recursive=True,
            recursive_depth=3,

            returns_split_count=True,

            n_jobs=1,
            mp_context="fork",
            max_threads_per_process=1,
            progress_bar=True,
        )

        merge_radius_um = params["clustering"]["merge_radius_um"]

        post_merge_label = merge_clusters(
            peaks,
            post_split_label,
            recording,
            features_folder,
            radius_um=merge_radius_um,
            
            method="waveforms_lda",
            method_kwargs=dict(
                # neighbours_mask=neighbours_mask,
                waveforms_sparse_mask=sparse_mask,
                
                # feature_name="sparse_tsvd",
                feature_name="sparse_wfs",
                
                # projection='lda',
                projection='centroid',

                # criteria='diptest',
                # threshold_diptest=0.5,
                criteria="percentile",
                threshold_percentile=80.,
                
                # num_shift=0
                num_shift=2,
                
                ),

            n_jobs=10,
            mp_context="fork",
            max_threads_per_process=1,
            progress_bar=True,
            # progress_bar=False,
        )
            
        sparse_wfs = np.load(features_folder / "sparse_wfs.npy", mmap_mode="r")



        # labels_set = np.unique(post_merge_label)
        # labels_set = labels_set[labels_set >= 0]
        # mask = post_merge_label >= 0
        # templates = compute_template_from_sparse(peaks[mask], post_merge_label[mask], labels_set,
        #                                          sparse_wfs, sparse_mask, num_chans)
        # matching_params = params["matching"].copy()
        # matching_params = 
        # spikes = find_spikes_from_templates(
        #     recording, method="wobble", method_kwargs=matching_params, **job_kwargs
        # )


        labels_set = np.unique(post_merge_label)
        labels_set = labels_set[labels_set >= 0]
        mask = post_merge_label >= 0

        sorting_temp = NumpySorting.from_times_labels(
            peaks["sample_index"][mask], post_merge_label[mask], sampling_frequency,
            unit_ids=labels_set,
        )
        sorting_temp = sorting_temp.save(folder=sorter_output_folder / "sorting_temp")

        ms_before = params["templates"]["ms_before"]
        ms_after = params["templates"]["ms_after"]
        max_spikes_per_unit = 300

        we = extract_waveforms(
            recording, sorting_temp, sorter_output_folder / "waveforms_temp", ms_before=ms_before, ms_after=ms_after,
            max_spikes_per_unit=max_spikes_per_unit, **job_kwargs
        )

        matching_params = params["matching"].copy()
        matching_params["waveform_extractor"] = we
        matching_params["noise_levels"] = noise_levels
        matching_params["peak_sign"] = params["detection"]["peak_sign"]
        matching_params["detect_threshold"] = params["detection"]["detect_threshold"]
        matching_params["radius_um"] = params["detection"]["radius_um"]

        spikes = find_spikes_from_templates(
            recording, method="tridesclous", method_kwargs=matching_params, **job_kwargs
        )


        if params["save_array"]:
            np.save(sorter_output_folder / 'all_peaks.npy', all_peaks)
            np.save(sorter_output_folder / 'post_split_label.npy', post_split_label)
            np.save(sorter_output_folder / 'split_count.npy', split_count)
            np.save(sorter_output_folder / 'post_merge_label.npy', post_split_label)
            np.save(sorter_output_folder / 'spikes.npy', spikes)



        # TODO multi segments
        sorting = NumpySorting.from_times_labels(spikes["sample_index"], spikes["cluster_index"], sampling_frequency)
        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting



# TODO remove this when extrac waveforms to single buffer is merge
from spikeinterface.core import extract_waveforms_to_buffers
def extract_best_channel_waveform_chan(rec, peaks,
                                       ms_before=0.5, ms_after=1.5,
                                       **job_kwargs):
    n = rec.get_num_channels()
    unit_ids = np.arange(n, dtype='int64')
    sparsity_mask = np.eye(n, dtype='bool')
    
    spikes = np.zeros(peaks.size, dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")])
    spikes["sample_index"] = peaks["sample_index"]
    spikes["unit_index"] = peaks["channel_index"]
    spikes["segment_index"] = peaks["segment_index"]

    nbefore = int(ms_before * rec.sampling_frequency / 1000.)
    nafter = int(ms_after * rec.sampling_frequency/ 1000.)

    wfs_arrays = extract_waveforms_to_buffers(rec, spikes, unit_ids, nbefore, nafter,
                                              mode="shared_memory", return_scaled=False,
                                              sparsity_mask=sparsity_mask, copy=True,
                                              **job_kwargs,
                                              )
    
    all_wfs = np.concatenate([wfs for wfs in wfs_arrays.values()], axis=0)

    return all_wfs