import pytest


# from spikeinterface.extractors import read_spykingcircus


@pytest.mark.skip("SpykingCIRCUS can be tested after running run_spykingcircus()")
def test_spykingcircusextractors():
    # not tested here, tested in run_spykingcircus(...)
    pass

    #  sc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/spykingcircus_output/'
    # sorting = read_spykingcircus(sc_folder)
    #  print(sorting)


if __name__ == "__main__":
    test_spykingcircusextractors()
