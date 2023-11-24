from .si_based import ComponentsBasedSorter

import os
import shutil
import numpy as np

from spikeinterface.core import (
    NumpySorting,
    load_extractor,
    BaseRecording,
    get_noise_levels,
    extract_waveforms,
    get_channel_distances,
)
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.preprocessing import common_reference, zscore, whiten, highpass_filter

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False


class Spykingcircus2Sorter(ComponentsBasedSorter):
    sorter_name = "spykingcircus2"

    _default_params = {
        "general": {"ms_before": 2, "ms_after": 2, "radius_um": 100},
        "waveforms": {
            "max_spikes_per_unit": 200,
            "overwrite": True,
            "sparse": True,
            "method": "energy",
            "threshold": 0.25,
        },
        "filtering": {"freq_min": 150, "dtype": "float32"},
        "detection": {"peak_sign": "neg", "detect_threshold": 4},
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 20000},
        "clustering": {"legacy": False},
        "matching": {},
        "apply_preprocessing": True,
        "shared_memory": True,
        "matched_filtering": False,
        "job_kwargs": {"n_jobs": 0.8},
        "debug": False,
    }

    handle_multi_segment = True

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        assert HAVE_HDBSCAN, "spykingcircus2 needs hdbscan to be installed"

        # this is importanted only on demand because numba import are too heavy
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["verbose"] = verbose
        job_kwargs["progress_bar"] = verbose

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        sampling_rate = recording.get_sampling_frequency()
        num_channels = recording.get_num_channels()
        ms_before = params["general"]["ms_before"]
        ms_after = params["general"]["ms_after"]
        radius_um = params["general"]["radius_um"]

        ## First, we are filtering the data
        filtering_params = params["filtering"].copy()
        if params["apply_preprocessing"]:
            recording_f = highpass_filter(recording, **filtering_params)
            recording_f = common_reference(recording_f)
        else:
            recording_f = recording
            recording_f.annotate(is_filtered=True)

        recording_f = zscore(recording_f, dtype="float32")
        noise_levels = np.ones(num_channels, dtype=np.float32)

        ## Then, we are detecting peaks with a locally_exclusive method
        detection_params = params["detection"].copy()
        detection_params["noise_levels"] = noise_levels
        if "exclude_sweep_ms" not in detection_params:
            detection_params["exclude_sweep_ms"] = max(ms_before, ms_after)
        if "radius_um" not in detection_params:
            detection_params["radius_um"] = radius_um

        fs = recording_f.get_sampling_frequency()
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        num_samples = nbefore + nafter
        num_chans = recording_f.get_num_channels()

        peaks = detect_peaks(recording_f, "locally_exclusive", **detection_params, **job_kwargs)

        if params["matched_filtering"]:
            few_peaks = select_peaks(peaks, method="uniform", n_peaks=5000)
            few_wfs = extract_waveform_at_max_channel(recording_f, few_peaks, ms_before, ms_after, **job_kwargs)

            nbefore = int(sampling_rate * ms_before / 1000)
            prototype = few_wfs[:, :, 0]
            prototype = np.median(prototype, 0)
            detection_params["prototype"] = prototype
            peaks = detect_peaks(recording_f, "locally_exclusive_mf", **detection_params, **job_kwargs)

        if verbose:
            print("We found %d peaks in total" % len(peaks))

        ## We subselect a subset of all the peaks, by making the distributions os SNRs over all
        ## channels as flat as possible
        selection_params = params["selection"]
        selection_params["n_peaks"] = params["selection"]["n_peaks_per_channel"] * num_channels
        selection_params["n_peaks"] = max(selection_params["min_n_peaks"], selection_params["n_peaks"])

        selection_params.update({"noise_levels": noise_levels})
        selected_peaks = select_peaks(
            peaks, method="smart_sampling_amplitudes", select_per_channel=False, **selection_params
        )

        if verbose:
            print("We kept %d peaks for clustering" % len(selected_peaks))

        ## We launch a clustering (using hdbscan) relying on positions and features extracted on
        ## the fly from the snippets
        clustering_params = params["clustering"].copy()
        clustering_params["waveforms"] = params["waveforms"].copy()

        for k in ["ms_before", "ms_after"]:
            clustering_params["waveforms"][k] = params["general"][k]

        clustering_params.update(dict(shared_memory=params["shared_memory"]))
        clustering_params["job_kwargs"] = job_kwargs
        clustering_params["tmp_folder"] = sorter_output_folder / "clustering"
        clustering_params.update({"noise_levels": noise_levels})

        if "legacy" in clustering_params:
            legacy = clustering_params.pop("legacy")
        else:
            legacy = False

        if legacy:
            clustering_method = "circus"
        else:
            clustering_method = "random_projections"

        labels, peak_labels = find_cluster_from_peaks(
            recording_f, selected_peaks, method=clustering_method, method_kwargs=clustering_params
        )

        ## We get the labels for our peaks
        mask = peak_labels > -1
        sorting = NumpySorting.from_times_labels(
            selected_peaks["sample_index"][mask], peak_labels[mask].astype(int), sampling_rate
        )
        clustering_folder = sorter_output_folder / "clustering"
        clustering_folder.mkdir(parents=True, exist_ok=True)

        if not params["debug"]:
            shutil.rmtree(clustering_folder)
        else:
            np.save(clustering_folder / "labels", labels)
            np.save(clustering_folder / "peaks", selected_peaks)

        ## We get the templates our of such a clustering
        waveforms_params = params["waveforms"].copy()

        for k in ["ms_before", "ms_after"]:
            waveforms_params[k] = params["general"][k]

        if params["shared_memory"] and not params["debug"]:
            mode = "memory"
            waveforms_folder = None
        else:
            sorting = sorting.save(folder=clustering_folder)
            mode = "folder"
            waveforms_folder = sorter_output_folder / "waveforms"

        we = extract_waveforms(
            recording_f,
            sorting,
            waveforms_folder,
            return_scaled=False,
            precompute_template=["median"],
            mode=mode,
            **waveforms_params,
            **job_kwargs
        )

        ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces
        matching_params = params["matching"].copy()
        matching_params["waveform_extractor"] = we

        matching_job_params = job_kwargs.copy()
        for value in ["chunk_size", "chunk_memory", "total_memory", "chunk_duration"]:
            if value in matching_job_params:
                matching_job_params.pop(value)

        matching_job_params["chunk_duration"] = "100ms"

        spikes = find_spikes_from_templates(
            recording_f, method="circus-omp-svd", method_kwargs=matching_params, **matching_job_params
        )

        if params["debug"]:
            fitting_folder = sorter_output_folder / "fitting"
            fitting_folder.mkdir(parents=True, exist_ok=True)
            np.save(fitting_folder / "spikes", spikes)

        if verbose:
            print("We found %d spikes" % len(spikes))

        ## And this is it! We have a spyking circus
        sorting = NumpySorting.from_times_labels(spikes["sample_index"], spikes["cluster_index"], sampling_rate)
        sorting_folder = sorter_output_folder / "sorting"

        if sorting_folder.exists():
            shutil.rmtree(sorting_folder)

        sorting = sorting.save(folder=sorting_folder)

        return sorting
