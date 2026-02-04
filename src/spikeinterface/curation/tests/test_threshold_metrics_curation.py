import pytest
import json

from spikeinterface.curation.tests.common import sorting_analyzer_for_curation
from spikeinterface.curation import threshold_metrics_label_units


@pytest.fixture
def sorting_analyzer_with_metrics(sorting_analyzer_for_curation):
    """A sorting analyzer with computed quality metrics."""

    sorting_analyzer = sorting_analyzer_for_curation
    sorting_analyzer.compute("quality_metrics")
    return sorting_analyzer


def test_threshold_metrics_label_units(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function."""
    sorting_analyzer = sorting_analyzer_with_metrics

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if (
            snr >= thresholds["snr"]["min"]
            and thresholds["firing_rate"]["min"] <= firing_rate <= thresholds["firing_rate"]["max"]
        ):
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_units_with_file(sorting_analyzer_with_metrics, tmp_path):
    """Test the `threshold_metrics_label_units` function with thresholds from a JSON file."""
    sorting_analyzer = sorting_analyzer_with_metrics

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1},
    }

    thresholds_file = tmp_path / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f)

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds_file,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if snr >= thresholds["snr"]["min"] and firing_rate >= thresholds["firing_rate"]["min"]:
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_units_with_external_metrics(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function with external metrics DataFrame."""
    sorting_analyzer = sorting_analyzer_with_metrics
    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    qm = sorting_analyzer.get_extension("quality_metrics").get_data()

    labels = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=qm,
        thresholds=thresholds,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if (
            snr >= thresholds["snr"]["min"]
            and thresholds["firing_rate"]["min"] <= firing_rate <= thresholds["firing_rate"]["max"]
        ):
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_external_labels(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function with custom pass/fail labels."""
    sorting_analyzer = sorting_analyzer_with_metrics
    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds=thresholds,
        pass_label="accepted",
        fail_label="rejected",
    )
    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)
    assert set(labels["label"]).issubset({"accepted", "rejected"})
