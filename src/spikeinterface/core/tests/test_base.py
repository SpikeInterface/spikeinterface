"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""
import pytest
from typing import Sequence
from pathlib import Path

from spikeinterface.core.base import BaseExtractor
from spikeinterface.core import generate_recording, concatenate_recordings
from spikeinterface.core.core_tools import dict_contains_extractors
from spikeinterface.core.testing import check_recordings_equal


class DummyDictExtractor(BaseExtractor):
    def __init__(self, main_ids: Sequence, base_dicts=None) -> None:
        super().__init__(main_ids)

        self._kwargs = dict(base_dicts=base_dicts)


def generate():
    return generate_recording(seed=0, durations=[2])


@pytest.fixture
def recording():
    return generate()


def make_nested_extractors(extractor):
    extractor_wih_parent = extractor.frame_slice(start_frame=0, end_frame=100)
    extractor_with_parent_list = concatenate_recordings([extractor, extractor])
    extractor_with_parent_list_with_parents = concatenate_recordings(
        [extractor_with_parent_list, extractor_with_parent_list]
    )
    extractor_with_parent_dict = DummyDictExtractor(
        main_ids=extractor._main_ids, base_dicts=dict(a=extractor, b=extractor, c=extractor)
    )
    return (
        extractor_wih_parent,
        extractor_with_parent_list,
        extractor_with_parent_list_with_parents,
        extractor_with_parent_dict,
    )


def test_check_if_dumpable(recording):
    test_extractor = recording

    # make a list of dumpable objects
    extractors_dumpable = make_nested_extractors(test_extractor)
    for extractor in extractors_dumpable:
        assert extractor.check_if_dumpable()

    # make not dumpable
    test_extractor._is_dumpable = False
    extractors_not_dumpable = make_nested_extractors(test_extractor)
    for extractor in extractors_not_dumpable:
        assert not extractor.check_if_dumpable()


def test_check_if_json_serializable(recording):
    test_extractor = recording

    # make a list of dumpable objects
    test_extractor._is_json_serializable = True
    extractors_json_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_json_serializable:
        print(extractor)
        assert extractor.check_if_json_serializable()

    # make not dumpable
    test_extractor._is_json_serializable = False
    extractors_not_json_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_not_json_serializable:
        print(extractor)
        assert not extractor.check_if_json_serializable()


def test_to_dict(recording):
    d0 = recording.to_dict()
    d0_recursive = recording.to_dict(recursive=True)
    assert not dict_contains_extractors(d0)
    assert not dict_contains_extractors(d0_recursive)

    nested_extractors = make_nested_extractors(recording)
    for extractor in nested_extractors:
        d1 = extractor.to_dict()
        d1_recursive = extractor.to_dict(recursive=True)

        assert dict_contains_extractors(d1)
        assert not dict_contains_extractors(d1_recursive)


def test_relative_to(recording, tmp_path):
    recording_saved = recording.save(folder=tmp_path / "test")
    folder_path = Path(recording_saved._kwargs["folder_path"])
    relative_folder = tmp_path.parent

    d1 = recording_saved.to_dict(recursive=True)
    d2 = recording_saved.to_dict(recursive=True, relative_to=relative_folder)

    assert d1["kwargs"]["folder_path"] == str(folder_path.absolute())
    assert d2["kwargs"]["folder_path"] != str(folder_path.absolute())
    assert d2["kwargs"]["folder_path"] == str(folder_path.relative_to(relative_folder))
    assert (
        str((relative_folder / Path(d2["kwargs"]["folder_path"])).resolve().absolute()) == d1["kwargs"]["folder_path"]
    )

    recording_loaded = BaseExtractor.from_dict(d2, base_folder=relative_folder)
    check_recordings_equal(recording_saved, recording_loaded, return_scaled=False)

    # test double pass in memory
    recording_nested = recording_saved.channel_slice(recording_saved.channel_ids)
    d3 = recording_nested.to_dict(relative_to=relative_folder)
    recording_loaded2 = BaseExtractor.from_dict(d3, base_folder=relative_folder)
    check_recordings_equal(recording_nested, recording_loaded2, return_scaled=False)
    d4 = recording_nested.to_dict(relative_to=relative_folder)
    recording_loaded3 = BaseExtractor.from_dict(d4, base_folder=relative_folder)
    check_recordings_equal(recording_nested, recording_loaded3, return_scaled=False)

    # check that dump to json/pickle don't modify paths
    # full_folder_path = str(recording_saved._kwargs["folder_path"])


if __name__ == "__main__":
    recording = generate()
    test_check_if_dumpable(recording)
    test_check_if_json_serializable(recording)
    test_to_dict(recording)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        test_relative_to(recording, tmp_path)
