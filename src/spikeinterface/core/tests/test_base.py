"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""
from typing import Sequence
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core import generate_recording, concatenate_recordings


class DummyDictExtractor(BaseExtractor):
    def __init__(self, main_ids: Sequence, base_dicts=None) -> None:
        super().__init__(main_ids)

        self._kwargs = dict(base_dicts=base_dicts)


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


def test_check_if_memory_serializable():
    test_extractor = generate_recording(seed=0, durations=[2])

    # make a list of memory serializable objects
    extractors_mem_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_mem_serializable:
        assert extractor.check_if_memory_serializable()

    # make not not memory serilizable
    test_extractor._serializablility["memory"] = False
    extractors_not_mem_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_not_mem_serializable:
        assert not extractor.check_if_memory_serializable()


def test_check_if_serializable():
    test_extractor = generate_recording(seed=0, durations=[2])

    # make a list of json serializable objects
    test_extractor._serializablility["json"] = True
    extractors_json_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_json_serializable:
        print(extractor)
        assert extractor.check_serializablility("json")

    # make of not json serializable objects
    test_extractor._serializablility["json"] = False
    extractors_not_json_serializable = make_nested_extractors(test_extractor)
    for extractor in extractors_not_json_serializable:
        print(extractor)
        assert not extractor.check_serializablility("json")


if __name__ == "__main__":
    test_check_if_memory_serializable()
    test_check_if_serializable()
