from .curation_tools import find_duplicated_spikes

from .remove_redundant import remove_redundant_units, find_redundant_units
from .remove_duplicated_spikes import remove_duplicated_spikes
from .remove_excess_spikes import remove_excess_spikes
from .auto_merge import (
    compute_merge_unit_groups,
    auto_merge_units,
    get_potential_auto_merge,
)

# manual sorting,
from .curationsorting import CurationSorting, curation_sorting
from .mergeunitssorting import MergeUnitsSorting, merge_units_sorting
from .splitunitsorting import SplitUnitSorting, split_unit_sorting

# curation format
from .curation_format import validate_curation_dict, curation_label_to_dataframe, apply_curation, load_curation

from .sortingview_curation import apply_sortingview_curation

# automated curation
from .threshold_metrics_curation import threshold_metrics_label_units
from .model_based_curation import model_based_label_units, load_model, auto_label_units
from .train_manual_curation import train_model, get_default_classifier_search_spaces
from .unitrefine_curation import unitrefine_label_units
