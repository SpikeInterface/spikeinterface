from .drift_tools import (
    move_dense_templates,
    interpolate_templates,
    DriftingTemplates,
    InjectDriftingTemplatesRecording,
    make_linear_displacement,
)

from .hybrid_tools import (
    generate_hybrid_recording,
    estimate_templates_from_recording,
    select_templates,
    scale_template_to_range,
    relocate_templates,
)
from .noise_tools import generate_noise

from .drifting_generator import (
    make_one_displacement_vector,
    generate_displacement_vector,
    generate_drifting_recording,
)

from .template_database import (
    fetch_template_object_from_database,
    fetch_templates_database_info,
    list_available_datasets_in_template_database,
    query_templates_from_database,
)

# expose the core generate functions
from ..core.generate import (
    generate_recording,
    generate_sorting,
    generate_snippets,
    generate_templates,
    generate_recording_by_size,
    generate_ground_truth_recording,
    add_synchrony_to_sorting,
    synthesize_random_firings,
    inject_some_duplicate_units,
    inject_some_split_units,
    synthetize_spike_train_bad_isi,
    NoiseGeneratorRecording,
    noise_generator_recording,
    InjectTemplatesRecording,
    inject_templates,
)
