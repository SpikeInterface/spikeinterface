import os
import getpass

if getpass.getuser() == 'samuel':
    kilosort2_path = '/home/samuel/Documents/Spikeinterface/Kilosort2'
    os.environ["KILOSORT2_PATH"] = kilosort2_path

    kilosort_path = '/home/samuel/Documents/Spikeinterface/KiloSort/'
    os.environ["KILOSORT_PATH"] = kilosort_path

    ironclust_path = '/home/samuel/Documents/Spikeinterface/ironclust'
    os.environ["IRONCLUST_PATH"] = ironclust_path

    waveclust_path = '/home/samuel/Documents/Spikeinterface/wave_clus/'
    os.environ["WAVECLUS_PATH"] = waveclust_path

from spikeinterface.sorters import print_sorter_versions


def test_print_sorter_versions():
    print_sorter_versions()


if __name__ == '__main__':
    test_print_sorter_versions()
