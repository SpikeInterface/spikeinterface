from __future__ import annotations

import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import numpy as np
from spikeinterface.core.node_pipeline import (
    base_peak_dtype,
)
from spikeinterface.postprocessing.unit_locations import (
    dtype_localize_by_method,
)
import matplotlib.pyplot as plt
from load_kilosort_utils import compute_spike_amplitude_and_depth


recording, sorting = si.generate_ground_truth_recording(
    durations=[30.0],
    sampling_frequency=30000.0,
)
# job_kwargs = dict(n_jobs=2, chunk_size=10000, progress_bar=True)
job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)

if False:
    peaks_ = detect_peaks(
        recording, method="locally_exclusive", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1, **job_kwargs
    )

    list_locations = []

    peak_locations = localize_peaks(recording, peaks_, method="center_of_mass", **job_kwargs)
"""
dtype=[('sample_index', '<i8'), ('channel_index', '<i8'), ('amplitude', '<f8'), ('segment_index', '<i8')])

peaks (num_spikes,);
"""

"""
array([(9.47572745,  3.69015204), (8.84228701, 13.40429091),
       (4.7367672 ,  4.5717565 ), ..., (8.6549761 , 12.90780294),
       (4.12265426,  4.90699353), (9.81511981,  5.47056214)],
      dtype=[('x', '<f8'), ('y', '<f8')])
"""

sorter_output_path = (r"D:\data\New folder\CA_528_1\imec0_ks2",)

params = load_ks_dir(sorter_output_path, load_pcs=True, exclude_noise=exclude_noise)

spike_indexes, spike_amplitudes, weighted_locs, max_sites = compute_spike_amplitude_and_depth(
    r"D:\data\New folder\CA_528_1\imec0_ks2",
    localised_spikes_only=False,
    exclude_noise=True,
)


segment_indexes = np.zeros(spike_indexes.size, dtype=np.int64)

peaks = np.zeros(spike_indexes.size, dtype=base_peak_dtype)
peaks["sample_index"] = spike_indexes
peaks["channel_index"] = max_sites
peaks["amplitude"] = spike_amplitudes
peaks["segment_index"] = segment_indexes

# TODO: a test is to check this peaks has the same entries as the normal peaks!... ? ,aube not as its caught with the dtypwe
# TODO: locations can be NaN!

peak_locations = np.zeros(spike_indexes.size, dtype_localize_by_method["center_of_mass"])
peak_locations["x"] = weighted_locs[:, 0]
peak_locations["y"] = weighted_locs[:, 1]


is_nan = np.any(np.isnan(weighted_locs), axis=1)
peaks = np.delete(peaks, is_nan)
peak_locations = np.delete(peak_locations, is_nan)

plt.scatter(peak_locations["x"][::10], peak_locations["y"][::10], c=peaks["amplitude"][::10])
plt.show()
