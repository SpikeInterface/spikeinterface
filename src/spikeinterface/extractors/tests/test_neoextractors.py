import unittest
import platform
import subprocess
import os
from packaging import version

import pytest

from spikeinterface.core.testing import check_recordings_equal
from spikeinterface import get_global_dataset_folder
from spikeinterface.extractors import *

from spikeinterface.extractors.tests.common_tests import (
    RecordingCommonTestSuite,
    SortingCommonTestSuite,
    EventCommonTestSuite,
)

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))
local_folder = get_global_dataset_folder() / "ephy_testing_data"


def has_plexon2_dependencies():
    """
    Check if required Plexon2 dependencies are installed on different OS.
    """

    os_type = platform.system()

    if os_type == "Windows":
        # On Windows, no need for additional dependencies
        return True

    elif os_type == "Linux":
        # Check for 'wine' using dpkg
        try:
            result_wine = subprocess.run(
                ["dpkg", "-l", "wine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError:
            return False

        # Check for 'zugbruecke' using pip
        try:
            import zugbruecke

            return True
        except ImportError:
            return False
    else:
        # Not sure about MacOS
        raise ValueError(f"Unsupported OS: {os_type}")


class MearecRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecRecordingExtractor
    downloads = ["mearec"]
    entities = ["mearec/mearec_test_10s.h5"]
    neo_funcs = dict()


class MearecSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MEArecSortingExtractor
    downloads = ["mearec"]
    entities = ["mearec/mearec_test_10s.h5"]


class SpikeGLXRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = SpikeGLXRecordingExtractor
    downloads = ["spikeglx"]
    entities = [
        ("spikeglx/Noise4Sam_g0", {"stream_id": "imec0.ap"}),
        ("spikeglx/Noise4Sam_g0", {"stream_id": "imec0.ap", "load_sync_channel": True}),
        ("spikeglx/Noise4Sam_g0", {"stream_id": "imec0.lf"}),
        ("spikeglx/Noise4Sam_g0", {"stream_id": "nidq"}),
    ]


class OpenEphysBinaryRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysBinaryRecordingExtractor
    downloads = ["openephysbinary"]
    entities = [
        "openephysbinary/v0.4.4.1_with_video_tracking",
        ("openephysbinary/v0.5.3_two_neuropixels_stream", {"stream_id": "0"}),
        ("openephysbinary/v0.5.3_two_neuropixels_stream", {"stream_id": "1"}),
        ("openephysbinary/v0.5.x_two_nodes", {"stream_id": "0"}),
        ("openephysbinary/v0.5.x_two_nodes", {"stream_id": "1"}),
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"stream_id": "0", "block_index": 0}),
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"stream_id": "1", "block_index": 1}),
        (
            "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
            {"stream_id": "1", "block_index": 1, "load_sync_timestamps": True},
        ),
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"stream_id": "2", "block_index": 2}),
        (
            "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
            {"stream_id": "2", "block_index": 2, "load_sync_timestamps": True},
        ),
    ]


class OpenEphysBinaryEventTest(EventCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysBinaryEventExtractor
    downloads = ["openephysbinary"]
    entities = [
        "openephysbinary/v0.4.4.1_with_video_tracking",
        "openephysbinary/v0.5.3_two_neuropixels_stream",
        "openephysbinary/v0.5.x_two_nodes",
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"block_index": 0}),
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"block_index": 1}),
        ("openephysbinary/v0.6.x_neuropixels_multiexp_multistream", {"block_index": 2}),
    ]


class OpenEphysLegacyRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = OpenEphysLegacyRecordingExtractor
    downloads = ["openephys"]
    entities = [
        "openephys/OpenEphys_SampleData_1",
        # This has gaps!!!
        "openephys/OpenEphys_SampleData_2_(multiple_starts)",
    ]


class IntanRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = IntanRecordingExtractor
    downloads = ["intan"]
    entities = [
        ("intan/intan_rhd_test_1.rhd", {"stream_id": "0"}),
        ("intan/intan_rhd_test_1.rhd", {"stream_id": "2"}),
        ("intan/intan_rhd_test_1.rhd", {"stream_id": "3"}),
        ("intan/intan_rhs_test_1.rhs", {"stream_id": "0"}),
        ("intan/intan_rhs_test_1.rhs", {"stream_id": "3"}),
        ("intan/intan_rhs_test_1.rhs", {"stream_id": "4"}),
        ("intan/intan_rhs_test_1.rhs", {"stream_id": "11"}),
    ]


class IntanRecordingTestMultipleFilesFormat(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = IntanRecordingExtractor
    downloads = ["intan"]
    entities = [
        ("intan/intan_fpc_test_231117_052630/info.rhd", {"stream_name": "RHD2000 amplifier channel"}),
        ("intan/intan_fpc_test_231117_052630/info.rhd", {"stream_name": "RHD2000 auxiliary input channel"}),
        ("intan/intan_fpc_test_231117_052630/info.rhd", {"stream_name": "USB board ADC input channel"}),
        ("intan/intan_fpc_test_231117_052630/info.rhd", {"stream_name": "USB board digital input channel"}),
        ("intan/intan_fps_test_231117_052500/info.rhd", {"stream_name": "RHD2000 amplifier channel"}),
        ("intan/intan_fps_test_231117_052500/info.rhd", {"stream_name": "RHD2000 auxiliary input channel"}),
        ("intan/intan_fps_test_231117_052500/info.rhd", {"stream_name": "USB board ADC input channel"}),
        ("intan/intan_fps_test_231117_052500/info.rhd", {"stream_name": "USB board digital input channel"}),
    ]


class NeuroScopeRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuroScopeRecordingExtractor
    downloads = ["neuroscope"]
    entities = [
        "neuroscope/test1/test1.xml",
        ("neuroscope/test2/signal1.dat", {"xml_file_path": local_folder / "neuroscope" / "test2" / "recording.xml"}),
    ]


class NeuroExplorerRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuroExplorerRecordingExtractor
    downloads = ["neuroexplorer"]
    entities = [
        ("neuroexplorer/File_neuroexplorer_1.nex", {"stream_name": "ContChannel01"}),
        ("neuroexplorer/File_neuroexplorer_1.nex", {"stream_name": "ContChannel02"}),
        ("neuroexplorer/File_neuroexplorer_2.nex", {"stream_name": "ContChannel01"}),
        ("neuroexplorer/File_neuroexplorer_2.nex", {"stream_name": "ContChannel02"}),
    ]


class NeuroScopeSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuroScopeSortingExtractor
    downloads = ["neuroscope"]
    entities = [
        "neuroscope/dataset_1",
        {
            "resfile_path": local_folder / "neuroscope/dataset_1/YutaMouse42-15111710.res.1",
            "clufile_path": local_folder / "neuroscope/dataset_1/YutaMouse42-15111710.clu.1",
        },
    ]


class PlexonRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = PlexonRecordingExtractor
    downloads = ["plexon"]
    entities = [
        "plexon/File_plexon_3.plx",
    ]


class PlexonSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = PlexonSortingExtractor
    downloads = ["plexon"]
    entities = [
        ("plexon/File_plexon_1.plx"),
    ]


class NeuralynxRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuralynxRecordingExtractor
    downloads = ["neuralynx"]
    entities = [
        "neuralynx/Cheetah_v1.1.0/original_data",
        "neuralynx/Cheetah_v4.0.2/original_data",
        "neuralynx/Cheetah_v5.4.0/original_data",
        "neuralynx/Cheetah_v5.5.1/original_data",
        "neuralynx/Cheetah_v5.6.3/original_data",
        "neuralynx/Cheetah_v5.7.4/original_data",
    ]


class NeuralynxSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = NeuralynxSortingExtractor
    downloads = ["neuralynx"]
    entities = [
        "neuralynx/Cheetah_v5.5.1/original_data",
        "neuralynx/Cheetah_v5.6.3/original_data",
    ]


class BlackrockRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BlackrockRecordingExtractor
    downloads = ["blackrock"]
    entities = [
        "blackrock/FileSpec2.3001.ns5",
        ("blackrock/blackrock_2_1/l101210-001.ns2", {"stream_id": "2"}),
        "blackrock/blackrock_2_1/l101210-001.ns2",  # this also work now because the stream id is auto selected with suffix
        ("blackrock/blackrock_2_1/l101210-001.ns5", {"stream_id": "5"}),
    ]


class BlackrockSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BlackrockSortingExtractor
    downloads = ["blackrock"]
    entities = [
        "blackrock/FileSpec2.3001.nev",
        dict(file_path=local_folder / "blackrock/blackrock_2_1/l101210-001.nev", sampling_frequency=30_000.0),
    ]


class MCSRawRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MCSRawRecordingExtractor
    downloads = ["rawmcs"]
    entities = [
        "rawmcs/raw_mcs_with_header_1.raw",
    ]


class TdTRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = TdtRecordingExtractor
    downloads = ["tdt"]
    entities = [("tdt/aep_05", {"stream_id": "1"})]


class AxonaRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = AxonaRecordingExtractor
    downloads = ["axona"]
    entities = [
        "axona/axona_raw",
    ]


class KiloSortSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = KiloSortSortingExtractor
    downloads = ["phy"]
    entities = [
        "phy/phy_example_0",
    ]


class Spike2RecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = Spike2RecordingExtractor
    downloads = ["spike2/130322-1LY.smr"]
    entities = [
        ("spike2/130322-1LY.smr", {"stream_id": "1"}),
    ]


@pytest.mark.skipif(
    version.parse(platform.python_version()) >= version.parse("3.10"),
    reason="Sonpy only testing with Python < 3.10!",
)
class CedRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CedRecordingExtractor
    downloads = [
        "spike2/130322-1LY.smr",
        "spike2/m365_1sec.smrx",
    ]
    entities = [
        ("spike2/130322-1LY.smr", {"stream_id": "1"}),
        "spike2/m365_1sec.smrx",
    ]


class MaxwellRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = MaxwellRecordingExtractor
    downloads = ["maxwell"]
    entities = [
        "maxwell/MaxOne_data/Record/000011/data.raw.h5",
        (
            "maxwell/MaxTwo_data/Network/000028/data.raw.h5",
            {"stream_id": "well000", "rec_name": "rec0000", "install_maxwell_plugin": True},
        ),
    ]


class SpikeGadgetsRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = SpikeGadgetsRecordingExtractor
    downloads = ["spikegadgets"]
    entities = [
        ("spikegadgets/20210225_em8_minirec2_ac.rec", {"stream_id": "ECU"}),
        ("spikegadgets/20210225_em8_minirec2_ac.rec", {"stream_id": "trodes"}),
        "spikegadgets/W122_06_09_2019_1_fromSD.rec",
        "spikegadgets/SpikeGadgets_test_data_2xNpix1.0_20240318_173658.rec",
    ]


class BiocamRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = BiocamRecordingExtractor
    downloads = ["biocam/biocam_hw3.0_fw1.6.brw"]
    entities = ["biocam/biocam_hw3.0_fw1.6.brw"]


class AlphaOmegaRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = AlphaOmegaRecordingExtractor
    downloads = ["alphaomega"]
    entities = [
        "alphaomega/mpx_map_version4",
    ]


class AlphaOmegaEventTest(EventCommonTestSuite, unittest.TestCase):
    ExtractorClass = AlphaOmegaEventExtractor
    downloads = ["alphaomega"]
    entities = [
        "alphaomega/mpx_map_version4",
    ]


class EDFRecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = EDFRecordingExtractor
    downloads = ["edf"]
    entities = ["edf/edf+C.edf"]

    def test_pickling(self):
        """
        This test is skipped because EDFRecordingExtractor can't keep two references open
        See issue #1228.
        """
        pass


# We run plexon2 tests only if we have dependencies (wine)
@pytest.mark.skipif(not has_plexon2_dependencies(), reason="Required dependencies not installed")
class Plexon2RecordingTest(RecordingCommonTestSuite, unittest.TestCase):
    ExtractorClass = Plexon2RecordingExtractor
    downloads = ["plexon"]
    entities = [
        ("plexon/4chDemoPL2.pl2", {"stream_id": "3"}),
    ]


@pytest.mark.skipif(not has_plexon2_dependencies(), reason="Required dependencies not installed")
class Plexon2EventTest(EventCommonTestSuite, unittest.TestCase):
    ExtractorClass = Plexon2EventExtractor
    downloads = ["plexon"]
    entities = [
        ("plexon/4chDemoPL2.pl2"),
    ]


@pytest.mark.skipif(not has_plexon2_dependencies(), reason="Required dependencies not installed")
class Plexon2SortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = Plexon2SortingExtractor
    downloads = ["plexon"]
    entities = [("plexon/4chDemoPL2.pl2", {"sampling_frequency": 40000})]


if __name__ == "__main__":
    # test = MearecSortingTest()
    # test = SpikeGLXRecordingTest()
    # test = OpenEphysBinaryRecordingTest()
    # test = SpikeGLXRecordingTest()
    # test = OpenEphysBinaryRecordingTest()
    # test = OpenEphysLegacyRecordingTest()
    # test = CellExplorerSortingTest()
    # test = ItanRecordingTest()
    # test = EDFRecordingTest()
    # test = NeuroScopeRecordingTest()
    # test = PlexonRecordingTest()
    # test = PlexonSortingTest()
    # test = NeuralynxRecordingTest()
    test = Plexon2RecordingTest()
    # test = MCSRawRecordingTest()
    # test = KiloSortSortingTest()
    # test = Spike2RecordingTest()
    # test = CedRecordingTest()
    # test = MaxwellRecordingTest()
    # test = SpikeGadgetsRecordingTest()
    # test = NeuroScopeSortingTest()

    test.setUp()
    test.test_open()
    # test.test_pickling()
