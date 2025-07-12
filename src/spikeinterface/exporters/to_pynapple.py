from spikeinterface.core import SortingAnalyzer
import pandas as pd


def to_pynapple_TsGroup(
    sorting_analyzer: SortingAnalyzer,
    metadata: pd.DataFrame | None = None,
):
    """
    Returns a pynapple TsGroup object based on spike train data.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    metadata : pd.DataFrame | None, default: None
        Unit-based metdata to attach to TsGroup. Input Dataframe must have keys
        equal to the `unit_id`s of the `sorting_analyzer`.

    Returns
    -------
    spike_train_TsGroup : pynapple.TsGroup
        A TsGroup object from the pynapple package.
    """
    from pynapple import TsGroup, Ts

    unit_ids = sorting_analyzer.unit_ids
    spikes_trains = {
        unit_id: sorting_analyzer.sorting.get_unit_spike_train(unit_id=unit_id, return_times=True)
        for unit_id in unit_ids
    }

    spike_train_TsGroup = TsGroup(
        {int(unit_id): Ts(spike_train) for unit_id, spike_train in spikes_trains.items()},
        metadata=metadata,
    )

    return spike_train_TsGroup
