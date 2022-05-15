import numpy as np

def generate_channel_map(nchan, sample_rate, xcoords, ycoords, kcoords):
    """
    This function is to generate kilosort channel map

    Based on kilosort3_channelmap.m file, creates a python dict to be saved as
    .mat file and loaded in kilosort3_main.m

    Loading example in Matlab (shouldn't be assigned to a variable):
    >> load('/output_folder/chanMap.mat');

    Args:
        nchan (int): Number of channels
        sample_rate (float): Sample Rate
        xcoords (lists): List of xcoords
        ycoords (lists): List of ycoords
        kcoords (lists): List of kcoords

    Returns:
        channel_map (dict): dict with kilosort's channel_map variables data.
    """
    channel_map = {}
    channel_map['connected'] = np.full((nchan, 1), True)
    channel_map['chanMap0ind'] = np.arange(nchan)
    channel_map['chanMap'] = channel_map['chanMap0ind'] + 1

    channel_map['xcoords'] = np.array(xcoords).astype(float)
    channel_map['ycoords'] = np.array(ycoords).astype(float)
    channel_map['kcoords'] = np.array(kcoords).astype(float)

    channel_map['fs'] = float(sample_rate)
    return channel_map
