# This is kept in 0.97.0 and then will be removed
from .alignsorting import AlignSortingExtractor, align_sorting
from .correlograms import (
    CorrelogramsCalculator,
    compute_autocorrelogram_from_spiketrain,
    compute_correlograms,
    compute_correlograms_numba,
    compute_correlograms_numpy,
    compute_crosscorrelogram_from_spiketrain,
    correlogram_for_one_segment,
)
from .isi import (
    ISIHistogramsCalculator,
    compute_isi_histograms,
    compute_isi_histograms_from_spiketrain,
    compute_isi_histograms_numba,
    compute_isi_histograms_numpy,
)
from .noise_level import NoiseLevelsCalculator, compute_noise_levels
from .principal_component import (
    WaveformPrincipalComponent,
    compute_principal_components,
)
from .spike_amplitudes import SpikeAmplitudesCalculator, compute_spike_amplitudes
from .spike_locations import SpikeLocationsCalculator, compute_spike_locations
from .template_metrics import (
    TemplateMetricsCalculator,
    calculate_template_metrics,
    compute_template_metrics,
    get_template_metric_names,
)
from .template_similarity import (
    TemplateSimilarityCalculator,
    check_equal_template_with_distribution_overlap,
    compute_template_similarity,
)
from .template_tools import (
    get_template_amplitudes,
    get_template_channel_sparsity,
    get_template_extremum_amplitude,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
)
from .unit_localization import (
    UnitLocationsCalculator,
    compute_center_of_mass,
    compute_unit_locations,
)
