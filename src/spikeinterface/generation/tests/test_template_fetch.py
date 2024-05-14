import pytest
from spikeinterface.generation import fetch_templates_from_database
from spikeinterface.core.template import Templates


def test_basic_call():

    templates = fetch_templates_from_database()

    assert isinstance(templates, Templates)

    assert templates.num_units == 100
    assert templates.num_channels == 384
