import pytest


# from spikeinterface.extractors import read_hdsort


@pytest.mark.skip("HDSort can be tested after running run_hdsort()")
def test_hdsortextractors():
    # no tested here, tested un run_hdsort(...)
    pass

    # hdsort_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/hdsort_output/'
    #  hdsort_folder = '/home/samuel/Bureau/hdsort_output/hdsort_output/hdsort_output_results.mat'
    #  sorting = HDSortSortingExtractor(hdsort_folder)
    #  print(sorting)


if __name__ == "__main__":
    test_hdsortextractors()
