%  create a channel map file

Nchannels = {nchan}; % number of channels
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;

xcoords = {xcoords};
ycoords = {ycoords};
kcoords   = {kcoords};

fs = {sample_rate}; % sampling frequency
save(fullfile('chanMap.mat'), ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
