import numpy as np

from spikeinterface.core.template import Templates

from spikeinterface.generation import (
    fetch_template_object_from_database,
    fetch_templates_database_info,
    list_available_datasets_in_template_database,
    query_templates_from_database,
)


def test_fetch_template_object_from_database():

    available_datasets = list_available_datasets_in_template_database()
    assert len(available_datasets) > 0

    templates = fetch_template_object_from_database("test_templates.zarr")
    assert isinstance(templates, Templates)

    assert templates.num_units == 89
    assert templates.num_samples == 240
    assert templates.num_channels == 384


def test_fetch_templates_database_info():
    import pandas as pd

    templates_info = fetch_templates_database_info()

    assert isinstance(templates_info, pd.DataFrame)

    assert "dataset" in templates_info.columns


def test_query_templates_from_database():
    templates_info = fetch_templates_database_info()

    templates_info = templates_info.iloc[[1, 3, 5]]
    num_selected = len(templates_info)

    templates = query_templates_from_database(templates_info)

    assert isinstance(templates, Templates)

    assert templates.num_units == num_selected


if __name__ == "__main__":
    test_fetch_template_object_from_database()
    test_fetch_templates_database_info()
    test_query_templates_from_database()
