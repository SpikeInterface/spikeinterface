if __name__ != '__main__':
    import matplotlib

    matplotlib.use('Agg')

from spikeinterface import download_dataset
import spikeinterface.extractors as se

from spikeinterface.widgets.utils import get_unit_colors


def test_get_unit_colors():
    local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
    sorting = se.MEArecSortingExtractor(local_path)

    colors = get_unit_colors(sorting)
    print(colors)
    

if __name__ == '__main__':
    test_get_unit_colors()
