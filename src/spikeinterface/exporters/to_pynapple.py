from __future__ import annotations

from spikeinterface.core import SortingAnalyzer, BaseSorting
import numpy as np
from warnings import warn


def to_pynapple_tsgroup(
    sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting,
    attach_unit_metadata=True,
    segment_index=None,
):
    """
    Returns a pynapple TsGroup object based on spike train data.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer
        A SortingAnalyzer object
    attach_unit_metadata : bool, default: True
        If True, any relevant available metadata is attached to the TsGroup. Will attach
        `unit_locations`, `quality_metrics` and `template_metrics` if computed. If False,
        no metadata is included.
    segment_index : int | None, default: None
        The segment index. Can be None if mono-segment sorting.

    Returns
    -------
    spike_train_TsGroup : pynapple.TsGroup
        A TsGroup object from the pynapple package.
    """
    from pynapple import TsGroup, Ts
    import pandas as pd

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        sorting = sorting_analyzer_or_sorting.sorting
    elif isinstance(sorting_analyzer_or_sorting, BaseSorting):
        sorting = sorting_analyzer_or_sorting
    else:
        raise TypeError(
            f"The `sorting_analyzer_or_sorting` argument must be a SortingAnalyzer or Sorting object, not a {type(sorting_analyzer_or_sorting)} type object."
        )

    unit_ids = sorting.unit_ids

    unit_ids_castable = True
    try:
        unit_ids_ints = [int(unit_id) for unit_id in unit_ids]
    except ValueError:
        warn_msg = "Pynapple requires integer unit ids, but `unit_ids` cannot be cast to int. "
        warn_msg += "We will set the index of the TsGroup to [0,1,2,...] and attach the original "
        warn_msg += "unit ids to the TsGroup as metadata with the name 'unit_id'."
        warn(warn_msg)
        unit_ids_ints = np.arange(len(unit_ids))
        unit_ids_castable = False

    spikes_trains = {
        unit_id_int: sorting.get_unit_spike_train(unit_id=unit_id, return_times=True, segment_index=segment_index)
        for unit_id_int, unit_id in zip(unit_ids_ints, unit_ids)
    }

    metadata_list = []
    if not unit_ids_castable:
        metadata_list.append(pd.DataFrame(unit_ids, columns=["unit_id"]))

    # Look for good metadata to add, if there is a sorting analyzer
    if attach_unit_metadata and isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):

        metadata_list = []
        if (unit_locations := sorting_analyzer_or_sorting.get_extension("unit_locations")) is not None:
            array_of_unit_locations = unit_locations.get_data()
            n_dims = np.shape(sorting_analyzer_or_sorting.get_extension("unit_locations").get_data())[1]
            pd_of_unit_locations = pd.DataFrame(
                array_of_unit_locations, columns=["x", "y", "z"][:n_dims], index=unit_ids
            )
            metadata_list.append(pd_of_unit_locations)
        if (quality_metrics := sorting_analyzer_or_sorting.get_extension("quality_metrics")) is not None:
            metadata_list.append(quality_metrics.get_data())
        if (template_metrics := sorting_analyzer_or_sorting.get_extension("template_metrics")) is not None:
            metadata_list.append(template_metrics.get_data())

    if len(metadata_list) > 0:
        metadata = pd.concat(metadata_list, axis=1)
        metadata.index = unit_ids_ints
    else:
        metadata = None

    spike_train_tsgroup = TsGroup(
        {unit_id: Ts(spike_train) for unit_id, spike_train in spikes_trains.items()},
        metadata=metadata,
    )

    return spike_train_tsgroup
