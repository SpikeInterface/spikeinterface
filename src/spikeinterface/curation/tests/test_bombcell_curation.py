import pytest
from pathlib import Path
from spikeinterface.curation.tests.common import sorting_analyzer_for_curation, trained_pipeline_path
from spikeinterface.curation.bombcell_curation import bombcell_label_units


def test_bombcell_label_units(sorting_analyzer_for_curation):
    """Test bombcell_label_units function on a sorting_analyzer with computed quality metrics."""

    sorting_analyzer = sorting_analyzer_for_curation
    sorting_analyzer.compute("quality_metrics")
    sorting_analyzer.compute("template_metrics")

    unit_type, unit_type_string = bombcell_label_units(sorting_analyzer=sorting_analyzer)

    assert len(unit_type) == sorting_analyzer.unit_ids.size
    assert set(unit_type_string).issubset({"somatic", "non-somatic", "good", "mua", "noise"})
