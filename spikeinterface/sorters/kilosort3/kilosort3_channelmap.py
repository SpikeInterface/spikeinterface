import numpy as np
import scipy.io


def generate_channel_map_file(nchan, sample_rate, xcoords, ycoords, kcoords, output_folder):
    """
    This function generates kilosort3 channel map data and saves as `chanMap.mat`

    Based on kilosort3_channelmap.m file, creates a .mat file to be loaded in kilosort3_main.m

    Loading example in Matlab (shouldn't be assigned to a variable):
    >> load('/output_folder/chanMap.mat');

    Parameters
    ----------
        nchan: int
            Number of channels
        sample_rate: float
            Sample Rate
        xcoords: list
            List of xcoords
        ycoords: list
            List of ycoords
        kcoords: list
            List of kcoords
        output_folder: pathlib.Path
            Path object to save `ops.mat` file
    """
    channel_map = {}
    channel_map['connected'] = np.full((nchan, 1), True)
    channel_map['chanMap0ind'] = np.arange(nchan)
    channel_map['chanMap'] = channel_map['chanMap0ind'] + 1

    channel_map['xcoords'] = np.array(xcoords).astype(float)
    channel_map['ycoords'] = np.array(ycoords).astype(float)
    channel_map['kcoords'] = np.array(kcoords).astype(float)

    channel_map['fs'] = float(sample_rate)
    scipy.io.savemat(str(output_folder / 'chanMap.mat'), channel_map)
