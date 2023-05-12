import unittest

from spikeinterface.extractors.tests.common_tests import SortingCommonTestSuite
from spikeinterface.extractors.cellexplorersortingextractor import CellExplorerSortingExtractor
from spikeinterface.core.globals import get_global_dataset_folder


local_folder = get_global_dataset_folder() / 'ephy_testing_data'


class CellExplorerSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = CellExplorerSortingExtractor
    downloads = ['cellexplorer']
    entities = [
        'cellexplorer/dataset_1/20170311_684um_2088um_170311_134350.spikes.cellinfo.mat',
        ('cellexplorer/dataset_2/20170504_396um_0um_merge.spikes.cellinfo.mat', 
        {'session_info_matfile_path': 
            local_folder / 'cellexplorer/dataset_2/20170504_396um_0um_merge.sessionInfo.mat'}),
        "cellexplorer/dataset_3/20170519_864um_900um_merge.spikes.cellinfo.mat",
        
    ]
