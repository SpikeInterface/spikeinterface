import pytest
import numpy as np

from spikeinterface.qualitymetrics import (
    compute_pc_metrics,
)

from spikeinterface.qualitymetrics.pca_metrics import isolation_score_two_clusters


def test_calculate_pc_metrics(small_sorting_analyzer):
    import pandas as pd

    sorting_analyzer = small_sorting_analyzer
    res1 = compute_pc_metrics(sorting_analyzer, n_jobs=1, progress_bar=True, seed=1205)
    res1 = pd.DataFrame(res1)

    res2 = compute_pc_metrics(sorting_analyzer, n_jobs=2, progress_bar=True, seed=1205)
    res2 = pd.DataFrame(res2)

    for metric_name in res1.columns:
        if metric_name != "nn_unit_id":
            assert not np.all(np.isnan(res1[metric_name].values))
            assert not np.all(np.isnan(res2[metric_name].values))

        assert np.array_equal(res1[metric_name].values, res2[metric_name].values)


def test_isolation_two_clusters():

    # spikes with pca projections ,
    pcs = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])

    isolation_score = isolation_score_two_clusters(
        pcs=pcs, labels=labels, unit_id=0, other_unit_id=1, n_neighbors=2, seed=1205, max_spikes=1000
    )

    # isolation score is (1 + 1 + 1/2 + 1/2 + 1 + 1)/6 = 5/6
    assert isolation_score == 5 / 6
