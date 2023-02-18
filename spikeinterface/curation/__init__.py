from .auto_merge import get_potential_auto_merge
from .curation_tools import find_duplicated_spikes

# manual sorting,
from .curationsorting import CurationSorting
from .mergeunitssorting import MergeUnitsSorting
from .remove_duplicated_spikes import (
    RemoveDuplicatedSpikesSorting,
    remove_duplicated_spikes,
)
from .remove_redundant import find_redundant_units, remove_redundant_units
from .sortingview_curation import apply_sortingview_curation
from .splitunitsorting import SplitUnitSorting
