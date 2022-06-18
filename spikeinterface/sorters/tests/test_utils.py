import pytest

from spikeinterface.sorters import utils


@pytest.mark.parametrize('container_image, expected',
[
    ('tridesclous', 'tridesclous.sif'),
    ('tridesclous.sif', 'tridesclous.sif'),
    ('spikeinterface/tridesclous', 'tridesclous.sif'),
    ('spikeinterface/tridesclous:3.4.5', 'tridesclous_3.4.5.sif'),
    ('spyking-circus', 'spyking-circus.sif'),
    ('spyking-circus:1.2.3', 'spyking-circus_1.2.3.sif'),
    ('spyking-circus:latest', 'spyking-circus_latest.sif'),
])
def test_resolve_sif_file(container_image, expected):
    """Tests for utils.resolve_sif_file"""
    assert utils.resolve_sif_file(container_image) == expected
