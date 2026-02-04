import pytest
import json

from spikeinterface.curation.tests.common import sorting_analyzer_for_curation
from spikeinterface.curation import threshold_metrics_label_units


def test_threshold_metrics_label_units(sorting_analyzer_for_curation):
    """Test the `threshold_metrics_label_units` function."""
    sorting_analyzer_for_curation.compute("quality_metrics")

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    labels = threshold_metrics_label_units(
        sorting_analyzer_for_curation,
        thresholds,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer_for_curation.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer_for_curation.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer_for_curation.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if (
            snr >= thresholds["snr"]["min"]
            and thresholds["firing_rate"]["min"] <= firing_rate <= thresholds["firing_rate"]["max"]
        ):
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_units_with_file(sorting_analyzer_for_curation, tmp_path):
    """Test the `threshold_metrics_label_units` function with thresholds from a JSON file."""
    sorting_analyzer_for_curation.compute("quality_metrics")

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1},
    }

    thresholds_file = tmp_path / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f)

    labels = threshold_metrics_label_units(
        sorting_analyzer_for_curation,
        thresholds_file,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer_for_curation.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer_for_curation.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer_for_curation.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if snr >= thresholds["snr"]["min"] and firing_rate >= thresholds["firing_rate"]["min"]:
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"
