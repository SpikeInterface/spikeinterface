import pytest
import unittest

import numpy as np

from spikeinterface.extractors.extractor_classes import CompressedBinaryIblExtractor, read_cbin_ibl

from spikeinterface.extractors.cbin_ibl import (
    _get_saved_channel_indices,
    _parse_saved_channel_subset,
    _parse_spikeglx_meta_table,
    _read_cbin_probe,
    extract_stream_info,
)

from spikeinterface.extractors.tests.common_tests import RecordingCommonTestSuite, SortingCommonTestSuite


class CompressedBinaryIblExtractorTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CompressedBinaryIblExtractor
    downloads = []
    entities = []


# --------------------------------------------------------------------------------
# NP2.4 shank-split meta handling
#
# When IBL splits an NP2.4 recording into four single-shank recordings, the
# generated .meta file has two peculiarities:
#
#   * none of the list-valued fields carry the leading "~" that SpikeGLX writes,
#     so neo.rawio.spikeglxrawio.read_meta_file leaves them as raw strings;
#   * the true acquisition channels live in "snsSaveChanSubset_orig" while
#     "snsSaveChanSubset" only holds the local, post-split renumbering.
#
# probeinterface.read_spikeglx uses "snsSaveChanSubset" and therefore returns a
# silently wrong probe for these files (sync counted as a contact, contacts
# spread over the wrong shanks). Correcting that is what _read_cbin_probe does.
# These tests need only a .meta file, so they run without any downloaded data.
# --------------------------------------------------------------------------------

# 16 acquired channels in blocks of four, alternating shank 0 / shank 1, so the
# saved subset is non-contiguous exactly like a real 4-shank split. Channels
# 4-7 and 12-15 are the shank-1 channels, i.e. electrodes 0-7 on shank 1.
N_ACQUIRED_CHANNELS = 16
SYNC_CHANNEL_INDEX = N_ACQUIRED_CHANNELS
SAVED_CHANNEL_SUBSET_ORIG = "4:7,12:15,16"
EXPECTED_CHANNEL_NAMES = ["AP4", "AP5", "AP6", "AP7", "AP12", "AP13", "AP14", "AP15", "SY0"]


def _build_shank_split_meta_text(include_subset_orig=True):
    """Build a minimal NP2.4 shank-split .meta text, deliberately without any "~".

    Parameters
    ----------
    include_subset_orig : bool, default: True
        If False, omit "snsSaveChanSubset_orig" so the meta looks like a normal
        SpikeGLX file and _read_cbin_probe should defer to probeinterface.

    Returns
    -------
    str
        The full contents of the .meta file.
    """
    imro_entries, geom_entries, chan_map_entries = [], [], []
    for channel in range(N_ACQUIRED_CHANNELS):
        block, position = divmod(channel, 4)
        shank = block % 2
        electrode = (block // 2) * 4 + position
        # type-24 (NP2.4) IMRO entry: (channel shank bank refid electrode)
        imro_entries.append(f"({channel} {shank} 0 1 {electrode})")
        # NP2.4 geometry: two columns at 27/59 um from the shank edge, 15 um row pitch
        x_um = 27 if electrode % 2 == 0 else 59
        geom_entries.append(f"({shank}:{x_um}:{15 * (electrode // 2)}:1)")
        chan_map_entries.append(f"(AP{channel};{channel}:{channel})")
    chan_map_entries.append(f"(SY0;{SYNC_CHANNEL_INDEX}:{SYNC_CHANNEL_INDEX})")

    num_saved_channels = len(EXPECTED_CHANNEL_NAMES)
    fields = {
        "typeThis": "imec",
        "acqApLfSy": f"{N_ACQUIRED_CHANNELS},0,1",
        "snsApLfSy": f"{num_saved_channels - 1},0,1",
        "nSavedChans": str(num_saved_channels),
        # the local, post-split numbering that read_spikeglx wrongly relies on
        "snsSaveChanSubset": f"0:{num_saved_channels - 1}",
        "imSampRate": "30000",
        "fileSizeBytes": str(num_saved_channels * 2 * 3000),
        "imAiRangeMax": "0.5",
        "imMaxInt": "8192",
        "imChan0apGain": "80",
        "imDatPrb_type": "24",
        "imDatPrb_pn": "NP2010",
        "imDatPrb_sn": "20472319942",
        "imDatPrb_port": "1",
        "imDatPrb_slot": "3",
        "imroTbl": f"(24,{N_ACQUIRED_CHANNELS})" + "".join(imro_entries),
        "snsChanMap": f"({N_ACQUIRED_CHANNELS},0,1)" + "".join(chan_map_entries),
        "snsGeomMap": "(NP2010,4,250,70)" + "".join(geom_entries),
    }
    if include_subset_orig:
        fields["snsSaveChanSubset_orig"] = SAVED_CHANNEL_SUBSET_ORIG

    # NOTE: written without the leading "~" that SpikeGLX puts on list-valued
    # fields. That omission is the whole point of this fixture.
    return "".join(f"{key}={value}\n" for key, value in fields.items())


def _read_meta(meta_file):
    """Parse a .meta file the same way CompressedBinaryIblExtractor does."""
    from neo.rawio.spikeglxrawio import read_meta_file

    return read_meta_file(str(meta_file))


@pytest.fixture
def shank_split_meta_file(tmp_path):
    """Path to a minimal NP2.4 shank-split meta file."""
    meta_file = tmp_path / "np24_shank_split.ap.meta"
    meta_file.write_text(_build_shank_split_meta_text())
    return meta_file


def test_parse_spikeglx_meta_table_accepts_string_and_list():
    """The parser must cope with both the raw "~"-less string and a parsed list."""
    assert _parse_spikeglx_meta_table("(a)(b)(c)") == ["a", "b", "c"]
    assert _parse_spikeglx_meta_table(["a", "b", "c"]) == ["a", "b", "c"]


def test_parse_saved_channel_subset():
    """Ranges, singletons and the "all" sentinel."""
    assert _parse_saved_channel_subset("all") is None
    assert _parse_saved_channel_subset(None) is None
    np.testing.assert_array_equal(_parse_saved_channel_subset("0:3"), [0, 1, 2, 3])
    np.testing.assert_array_equal(_parse_saved_channel_subset("4:7,12:15,16"), [4, 5, 6, 7, 12, 13, 14, 15, 16])


def test_saved_channel_indices_prefer_orig(shank_split_meta_file):
    """snsSaveChanSubset_orig must take precedence over the local snsSaveChanSubset."""
    meta = _read_meta(shank_split_meta_file)
    np.testing.assert_array_equal(_get_saved_channel_indices(meta), [4, 5, 6, 7, 12, 13, 14, 15, 16])


def test_meta_without_tilde_stays_a_string(shank_split_meta_file):
    """Guards the premise of the fix: neo cannot list-parse these fields."""
    meta = _read_meta(shank_split_meta_file)
    assert isinstance(meta["imroTbl"], str)
    assert isinstance(meta["snsChanMap"], str)


def test_read_cbin_probe_selects_the_correct_shank(shank_split_meta_file):
    """The probe must hold only the 8 shank-1 contacts, at the right positions."""
    meta = _read_meta(shank_split_meta_file)
    probe = _read_cbin_probe(str(shank_split_meta_file), meta)

    assert probe.get_contact_count() == 8
    assert set(probe.shank_ids) == {"1"}
    # shank pitch 250 um, two columns 32 um apart, four rows at 15 um pitch
    np.testing.assert_array_equal(np.unique(probe.contact_positions[:, 0]), [250.0, 282.0])
    np.testing.assert_array_equal(np.unique(probe.contact_positions[:, 1]), [0.0, 15.0, 30.0, 45.0])
    np.testing.assert_array_equal(probe.device_channel_indices, np.arange(8))
    assert probe.annotations["serial_number"] == "20472319942"
    assert probe.annotations["part_number"] == "NP2010"


def test_read_cbin_probe_differs_from_plain_read_spikeglx(shank_split_meta_file):
    """Regression guard: plain read_spikeglx returns a wrong probe for this meta.

    It applies the local "snsSaveChanSubset" to the geometry table, so the sync
    channel becomes a contact and the contacts land on two different shanks.
    If probeinterface ever learns about "snsSaveChanSubset_orig" this test will
    fail, which is the signal that _read_cbin_probe can be removed.
    """
    import probeinterface

    meta = _read_meta(shank_split_meta_file)
    wrong_probe = probeinterface.read_spikeglx(str(shank_split_meta_file))
    correct_probe = _read_cbin_probe(str(shank_split_meta_file), meta)

    assert wrong_probe.get_contact_count() == 9  # 8 contacts + sync wrongly counted as a contact
    assert set(wrong_probe.shank_ids) == {"0", "1"}  # contacts wrongly spread over two shanks
    assert correct_probe.get_contact_count() != wrong_probe.get_contact_count()


def test_read_cbin_probe_falls_back_when_no_subset_orig(tmp_path):
    """A normal SpikeGLX meta must be passed through to probeinterface untouched."""
    import probeinterface

    meta_file = tmp_path / "plain.ap.meta"
    meta_file.write_text(_build_shank_split_meta_text(include_subset_orig=False))
    meta = _read_meta(meta_file)

    probe = _read_cbin_probe(str(meta_file), meta)
    reference_probe = probeinterface.read_spikeglx(str(meta_file))
    np.testing.assert_array_equal(probe.contact_positions, reference_probe.contact_positions)


def test_extract_stream_info_uses_the_original_subset(shank_split_meta_file):
    """Channel names must follow snsSaveChanSubset_orig, not the local numbering."""
    meta = _read_meta(shank_split_meta_file)
    info = extract_stream_info(str(shank_split_meta_file), meta)

    assert info["num_chan"] == len(EXPECTED_CHANNEL_NAMES)
    assert info["channel_names"] == EXPECTED_CHANNEL_NAMES
    assert info["channel_gains"].shape[0] == info["num_chan"]
    assert info["channel_offsets"].shape[0] == info["num_chan"]
    assert info["has_sync_trace"]


def test_extract_stream_info_raises_on_inconsistent_meta(tmp_path):
    """The nSavedChans consistency guard must fire on a corrupted meta."""
    meta_text = _build_shank_split_meta_text().replace("nSavedChans=9", "nSavedChans=42")
    meta_file = tmp_path / "bad.ap.meta"
    meta_file.write_text(meta_text)
    meta = _read_meta(meta_file)

    with pytest.raises(ValueError, match="does not match nSavedChans"):
        extract_stream_info(str(meta_file), meta)


# ~ def test_read_cbin_ibl():
# ~ base_folder = '/media/samuel/dataspikesorting/DataSpikeSorting/olivier_destripe/'
# ~ data_folder = base_folder + '4c04120d-523a-4795-ba8f-49dbb8d9f63a'
# ~ rec = read_cbin_ibl(data_folder)

# ~ import matplotlib.pyplot as plt
# ~ import spikeinterface.widgets as sw
# ~ from probeinterface.plotting import plot_probe
# ~ sw.plot_traces(rec)
# ~ plot_probe(rec.get_probe())
# ~ plt.show()


if __name__ == "__main__":
    # ~ test_read_cbin_ibl()

    test = CompressedBinaryIblExtractorTest()
    test.setUp()
    test.test_open()
