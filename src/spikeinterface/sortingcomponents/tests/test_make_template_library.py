from pathlib import Path

from spikeinterface.core.template import Templates
from spikeinterface.sortingcomponents.tests.common import make_dataset
from spikeinterface.sortingcomponents.tools import estimate_templates_from_recording


def test_make_template_library(tmp_path):
    """Tests that the `estimate_templates_from_recording` does return a Templates object."""
    recording, _ = make_dataset()
    recording = recording.save(folder=tmp_path / "generated_recording")

    temp_folder = Path("my_template_temp_folder")
    template_library = estimate_templates_from_recording(recording, temp_folder=temp_folder)

    # ensure we get templates
    assert isinstance(template_library, Templates)

    # and the intermediate directory has been deleted
    assert not temp_folder.is_dir()
