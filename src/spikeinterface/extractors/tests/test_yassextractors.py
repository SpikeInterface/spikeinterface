import pytest


# from spikeinterface.extractors import read_yass


@pytest.mark.skip("YASS can be tested after running run_yass()")
def test_yassextractors():
    # not tested here, tested in run_yass(...)
    pass

    #  yass_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/yass_output/'
    #  sorting = read_yass(yass_folder)
    #  print(sorting)


if __name__ == "__main__":
    test_yassextractors()
