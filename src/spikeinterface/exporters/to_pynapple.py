from spikeinterface.core import SortingAnalyzer, BaseSorting
import numpy as np


def to_pynapple_tsgroup(
    sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting,
    metadata=None,
):
    """
    Returns a pynapple TsGroup object based on spike train data.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer
        A SortingAnalyzer object
    metadata : pd.DataFrame | dict | None, default: None
        Metadata associated with each unit. Metadata names are pulled from DataFrame columns
        or dictionary keys. The length of the metadata should match the number of units.

    Returns
    -------
    spike_train_TsGroup : pynapple.TsGroup
        A TsGroup object from the pynapple package.
    """
    from pynapple import TsGroup, Ts

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        sorting = sorting_analyzer_or_sorting.sorting
    elif isinstance(sorting_analyzer_or_sorting, BaseSorting):
        sorting = sorting_analyzer_or_sorting
    else:
        raise TypeError(
            f"The `sorting_analyzer_or_sorting` argument must be a SortingAnalyzer or Sorting object, not a {type(sorting_analyzer_or_sorting)} type object."
        )

    unit_ids = sorting.unit_ids
    spikes_trains = {unit_id: sorting.get_unit_spike_train(unit_id=unit_id, return_times=True) for unit_id in unit_ids}

    # Look for good metadata to add, if there is a sorting analyzer
    if metadata is None and isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):

        import pandas as pd

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

            # pynapple requires integer indices
            metadata.index = metadata.index.astype("int")

    spike_train_tsgroup = TsGroup(
        {int(unit_id): Ts(spike_train) for unit_id, spike_train in spikes_trains.items()},
        metadata=metadata,
    )

    return spike_train_tsgroup
