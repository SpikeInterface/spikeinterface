import pytest

# from spikeinterface.extractors import read_klusta


@pytest.mark.skip("Klusta can be tested after running run_klusta()")
def test_klustaextractors():
    # no tested here, tested un run_klusta
    pass

    #  klusta_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/klusta_output/'
    #  sorting = read_klusta(klusta_folder)
    #  print(sorting)


if __name__ == "__main__":
    test_klustaextractors()
