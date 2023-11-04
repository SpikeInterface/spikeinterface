# """Sorting components: clustering"""
from pathlib import Path

import shutil
import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

import random, string, os
from spikeinterface.core import get_global_tmp_folder, get_noise_levels, get_channel_distances
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching, remove_duplicates_via_dip
from spikeinterface.core import NumpySorting
from spikeinterface.core import extract_waveforms
from spikeinterface.core.recording_tools import get_channel_distances, get_random_data_chunks


class CircusClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "peak_locations": None,
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "hdbscan_kwargs": {
            "min_cluster_size": 50,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
        },
        "cleaning_kwargs": {},
        "tmp_folder": None,
        "radius_um": 100,
        "n_pca": 10,
        "max_spikes_per_unit": 200,
        "ms_before": 1.5,
        "ms_after": 2.5,
        "cleaning_method": "dip",
        "waveform_mode": "memmap",
        "job_kwargs": {"n_jobs": -1, "chunk_memory": "10M"},
    }

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()

        tmp_folder = params["tmp_folder"]
        if params["waveform_mode"] == "memmap":
            if tmp_folder is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))
            else:
                tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir()
            params2["tmp_folder"] = tmp_folder
        elif params["waveform_mode"] == "shared_memory":
            assert tmp_folder is None, "tmp_folder must be None for shared_memory"
        else:
            raise ValueError("'waveform_mode' must be 'memmap' or 'shared_memory'")

        return params2

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "twisted clustering needs hdbscan to be installed"

        params = cls._check_params(recording, peaks, params)
        d = params

        if d["peak_locations"] is None:
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks

            peak_locations = localize_peaks(recording, peaks, **d["peak_localization_kwargs"], **d["job_kwargs"])
        else:
            peak_locations = d["peak_locations"]

        tmp_folder = d["tmp_folder"]
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)

        location_keys = ["x", "y"]
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)

        chan_locs = recording.get_channel_locations()

        peak_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
        spikes = np.zeros(peaks.size, dtype=peak_dtype)
        spikes["sample_index"] = peaks["sample_index"]
        spikes["segment_index"] = peaks["segment_index"]
        spikes["unit_index"] = peaks["channel_index"]

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((peaks.size, num_chans), dtype="bool")

        unit_inds = range(num_chans)
        chan_distances = get_channel_distances(recording)

        for main_chan in unit_inds:
            (closest_chans,) = np.nonzero(chan_distances[main_chan, :] <= params["radius_um"])
            sparsity_mask[main_chan, closest_chans] = True

        if params["waveform_mode"] == "shared_memory":
            wf_folder = None
        else:
            assert params["tmp_folder"] is not None, "tmp_folder must be supplied"
            wf_folder = params["tmp_folder"] / "sparse_snippets"
            wf_folder.mkdir()

        fs = recording.get_sampling_frequency()
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)
        num_samples = nbefore + nafter

        wfs_arrays = extract_waveforms_to_buffers(
            recording,
            spikes,
            unit_inds,
            nbefore,
            nafter,
            mode=params["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=recording.get_dtype(),
            sparsity_mask=sparsity_mask,
            copy=(params["waveform_mode"] == "shared_memory"),
            **params["job_kwargs"],
        )

        n_loc = len(location_keys)
        import sklearn.decomposition, hdbscan

        noise_levels = get_noise_levels(recording, return_scaled=False)

        nb_clusters = 0
        peak_labels = np.zeros(len(spikes), dtype=np.int32)

        noise = get_random_data_chunks(
            recording,
            return_scaled=False,
            num_chunks_per_segment=params["max_spikes_per_unit"],
            chunk_size=nbefore + nafter,
            concatenated=False,
            seed=None,
        )
        noise = np.stack(noise, axis=0)

        for main_chan, waveforms in wfs_arrays.items():
            idx = np.where(spikes["unit_index"] == main_chan)[0]
            (channels,) = np.nonzero(sparsity_mask[main_chan])
            sub_noise = noise[:, :, channels]

            if len(waveforms) > 0:
                sub_waveforms = waveforms

                wfs = np.swapaxes(sub_waveforms, 1, 2).reshape(len(sub_waveforms), -1)
                noise_wfs = np.swapaxes(sub_noise, 1, 2).reshape(len(sub_noise), -1)

                n_pca = min(d["n_pca"], len(wfs))
                pca = sklearn.decomposition.PCA(n_pca)

                hdbscan_data = np.vstack((wfs, noise_wfs))

                pca.fit(wfs)
                hdbscan_data_pca = pca.transform(hdbscan_data)
                clustering = hdbscan.hdbscan(hdbscan_data_pca, **d["hdbscan_kwargs"])

                noise_labels = clustering[0][len(wfs) :]
                valid_labels = clustering[0][: len(wfs)]

                shared_indices = np.intersect1d(np.unique(noise_labels), np.unique(valid_labels))
                for l in shared_indices:
                    idx_noise = noise_labels == l
                    idx_valid = valid_labels == l
                    if np.sum(idx_noise) > np.sum(idx_valid):
                        valid_labels[idx_valid] = -1

                if np.unique(valid_labels).min() == -1:
                    valid_labels += 1

                for l in np.unique(valid_labels):
                    idx_valid = valid_labels == l
                    if np.sum(idx_valid) < d["hdbscan_kwargs"]["min_cluster_size"]:
                        valid_labels[idx_valid] = -1

                peak_labels[idx] = valid_labels + nb_clusters

                labels = np.unique(valid_labels)
                labels = labels[labels >= 0]
                nb_clusters += len(labels)

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        best_spikes = {}
        nb_spikes = 0

        all_indices = np.arange(0, peak_labels.size)

        for unit_ind in labels:
            mask = peak_labels == unit_ind
            best_spikes[unit_ind] = np.random.permutation(all_indices[mask])[: params["max_spikes_per_unit"]]
            nb_spikes += best_spikes[unit_ind].size

        spikes = np.zeros(nb_spikes, dtype=peak_dtype)

        mask = np.zeros(0, dtype=np.int32)
        for unit_ind in labels:
            mask = np.concatenate((mask, best_spikes[unit_ind]))

        idx = np.argsort(mask)
        mask = mask[idx]

        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        if params["waveform_mode"] == "shared_memory":
            wf_folder = None
        else:
            assert params["tmp_folder"] is not None, "tmp_folder must be supplied"
            wf_folder = params["tmp_folder"] / "dense_snippets"
            wf_folder.mkdir()

        cleaning_method = params["cleaning_method"]

        print(f"We found {len(labels)} raw clusters, starting to clean with {cleaning_method}...")

        if cleaning_method == "cosine":
            wfs_arrays = extract_waveforms_to_buffers(
                recording,
                spikes,
                labels,
                nbefore,
                nafter,
                mode=params["waveform_mode"],
                return_scaled=False,
                folder=wf_folder,
                dtype=recording.get_dtype(),
                sparsity_mask=None,
                copy=(params["waveform_mode"] == "shared_memory"),
                **params["job_kwargs"],
            )

            labels, peak_labels = remove_duplicates(
                wfs_arrays, noise_levels, peak_labels, num_samples, num_chans, **params["cleaning_kwargs"]
            )

        elif cleaning_method == "dip":
            wfs_arrays = extract_waveforms_to_buffers(
                recording,
                spikes,
                labels,
                nbefore,
                nafter,
                mode=params["waveform_mode"],
                return_scaled=False,
                folder=wf_folder,
                dtype=recording.get_dtype(),
                sparsity_mask=None,
                copy=(params["waveform_mode"] == "shared_memory"),
                **params["job_kwargs"],
            )

            labels, peak_labels = remove_duplicates_via_dip(wfs_arrays, peak_labels)

        elif cleaning_method == "matching":
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))

            sorting = NumpySorting.from_times_labels(spikes["sample_index"], spikes["unit_index"], fs)
            we = extract_waveforms(
                recording,
                sorting,
                tmp_folder,
                overwrite=True,
                ms_before=params["ms_before"],
                ms_after=params["ms_after"],
                **params["job_kwargs"],
            )
            labels, peak_labels = remove_duplicates_via_matching(we, peak_labels, job_kwargs=params["job_kwargs"])
            shutil.rmtree(tmp_folder)

        print(f"We kept {len(labels)} non-duplicated clusters...")

        return labels, peak_labels
