import pytest


from spikeinterface.generation import (
    fetch_template_dataset,
    fetch_templates_info,
    list_available_datasets,
    get_templates_from_database,
)
from spikeinterface.core.template import Templates


def test_fetch_datasets():

    available_datasets = list_available_datasets()
    assert len(available_datasets) > 0

    templates = fetch_template_dataset("test_templates.zarr")
    assert isinstance(templates, Templates)

    assert templates.num_units == 100
    assert templates.num_channels == 384


def test_fetch_templates_info():
    import pandas as pd

    templates_info = fetch_templates_info()

    assert isinstance(templates_info, pd.DataFrame)

    assert "dataset" in templates_info.columns


def test_get_templates_from_database():
    templates_info = fetch_templates_info()

    templates_info = templates_info.iloc[::15]
    num_selected = len(templates_info)

    templates = get_templates_from_database(templates_info)

    assert isinstance(templates, Templates)

    assert templates.num_units == num_selected
