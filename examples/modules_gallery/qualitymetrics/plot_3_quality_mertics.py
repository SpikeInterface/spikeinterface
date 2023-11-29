"""
Quality Metrics Tutorial
========================

After spike sorting, you might want to validate the 'goodness' of the sorted units. This can be done using the
:code:`qualitymetrics` submodule, which computes several quality metrics of the sorted units.

"""

import spikeinterface.core as si
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import (compute_snrs, compute_firing_rates,
    compute_isi_violations, calculate_pc_metrics, compute_quality_metrics)

##############################################################################
# First, let's download a simulated dataset
# from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting = se.read_mearec(local_path)
print(recording)
print(sorting)

##############################################################################
# Extract spike waveforms
# -----------------------
#
# For convenience, metrics are computed on the :code:`WaveformExtractor` object,
# because it contains a reference to the "Recording" and the "Sorting" objects:

we = si.extract_waveforms(recording=recording,
                          sorting=sorting,
                          folder='waveforms_mearec',
                          sparse=False,
                          ms_before=1,
                          ms_after=2.,
                          max_spikes_per_unit=500,
                          n_jobs=1,
                          chunk_durations='1s')
print(we)

##############################################################################
# The :code:`spikeinterface.qualitymetrics` submodule has a set of functions that allow users to compute
# metrics in a compact and easy way. To compute a single metric, one can simply run one of the
# quality metric functions as shown below. Each function has a variety of adjustable parameters that can be tuned.

firing_rates = compute_firing_rates(we)
print(firing_rates)
isi_violation_ratio, isi_violations_count = compute_isi_violations(we)
print(isi_violation_ratio)
snrs = compute_snrs(we)
print(snrs)

##############################################################################
# Some metrics are based on the principal component scores, so they require a
# :code:`WaveformsPrincipalComponent` object as input:

pc = compute_principal_components(waveform_extractor=we, load_if_exists=True,
                                     n_components=3, mode='by_channel_local')
print(pc)

pc_metrics = calculate_pc_metrics(pc, metric_names=['nearest_neighbor'])
print(pc_metrics)

##############################################################################
# To compute more than one metric at once, we can use the :code:`compute_quality_metrics` function and indicate
# which metrics we want to compute. This will return a pandas dataframe:

metrics = compute_quality_metrics(we)
print(metrics)
