
# This is kept in 0.97.0 and then will be removed
from .template_tools import (
    get_template_amplitudes,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
    get_template_extremum_amplitude,
    get_template_channel_sparsity
)

from .template_metrics import (TemplateMetricsCalculator, compute_template_metrics,
                               calculate_template_metrics, get_template_metric_names)

from .template_similarity import (TemplateSimilarityCalculator,
                                  compute_template_similarity,
                                  check_equal_template_with_distribution_overlap)

from .principal_component import WaveformPrincipalComponent, compute_principal_components

from .spike_amplitudes import compute_spike_amplitudes, SpikeAmplitudesCalculator

from .correlograms import (CorrelogramsCalculator,
                           compute_autocorrelogram_from_spiketrain,
                           compute_crosscorrelogram_from_spiketrain,
                           compute_correlograms,
                           compute_correlograms_numba, compute_correlograms_numpy)

from .isi import (ISIHistogramsCalculator,
                  compute_isi_histograms_from_spiketrain, compute_isi_histograms,
                  compute_isi_histograms_numpy, compute_isi_histograms_numba)

from .spike_locations import compute_spike_locations, SpikeLocationsCalculator

from .unit_localization import (compute_unit_locations, UnitLocationsCalculator,
                                localize_units, compute_center_of_mass)

from .alignsorting import align_sorting, AlignSortingExtractor
