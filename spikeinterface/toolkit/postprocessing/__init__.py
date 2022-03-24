from .template_tools import (get_template_amplitudes,
                             get_template_extremum_channel,
                             get_template_extremum_channel_peak_shift,
                             get_template_extremum_amplitude,
                             get_template_channel_sparsity)

from .template_metrics import (calculate_template_metrics, get_template_metric_names)

from .template_similarity import compute_template_similarity, check_equal_template_with_distribution_overlap

from .principal_component import (WaveformPrincipalComponent,
                                  compute_principal_components)

from .spike_amplitudes import compute_spike_amplitudes, SpikeAmplitudesCalculator

from .correlograms import compute_autocorrelogram_from_spiketrain, compute_correlograms


from .unit_localization import localize_units, compute_center_of_mass
