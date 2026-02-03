import pytest
from pathlib import Path
from spikeinterface.curation.tests.common import sorting_analyzer_for_curation, trained_pipeline_path
from spikeinterface.curation.bombcell_curation import bombcell_label_units


@pytest.fixture
def sorting_analyzer_with_metrics(sorting_analyzer_for_curation):
    """A sorting analyzer with computed quality and template metrics."""

    sorting_analyzer = sorting_analyzer_for_curation
    sorting_analyzer.compute("quality_metrics")
    sorting_analyzer.compute("template_metrics")
    return sorting_analyzer


def test_bombcell_label_units(sorting_analyzer_with_metrics):
    """Test bombcell_label_units function on a sorting_analyzer with computed quality metrics."""

    sorting_analyzer = sorting_analyzer_with_metrics

    labels = bombcell_label_units(sorting_analyzer=sorting_analyzer)

    assert len(labels) == sorting_analyzer.unit_ids.size
    assert set(labels["label"]).issubset({"somatic", "non-somatic", "good", "mua", "noise"})


def test_bombcell_label_units_with_external_metrics(sorting_analyzer_with_metrics):
    """Test bombcell_label_units function with external metrics."""

    sorting_analyzer = sorting_analyzer_with_metrics

    # Create external metrics DataFrame
    import pandas as pd

    metrics_df = sorting_analyzer.get_metrics_extension_data()

    labels = bombcell_label_units(
        sorting_analyzer=None,
        external_metrics=metrics_df,
    )

    # run default metrics in analyzer
    labels2 = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
    )

    assert (labels["label_type"] == labels2["label_type"]).all()
    assert (labels["label"] == labels2["label"]).all()


def test_bombcell_label_units_with_threshold_file(sorting_analyzer_with_metrics, tmp_path):
    """Test bombcell_label_units function with thresholds from a JSON file."""

    sorting_analyzer = sorting_analyzer_with_metrics

    # Define custom thresholds
    custom_thresholds = {
        "snr": {"min": 5, "max": 100},
        "isi_violations": {"min": None, "max": 0.2},
    }

    # Save thresholds to a temporary JSON file
    import json

    threshold_file = tmp_path / "custom_thresholds.json"
    with open(threshold_file, "w") as f:
        json.dump(custom_thresholds, f)

    labels = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
        thresholds=threshold_file,
    )

    assert len(labels) == sorting_analyzer.unit_ids.size
