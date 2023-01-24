from pathlib import Path
import json 

import numpy as np
from scipy import signal
import torch 
from torch import nn
import torch.utils.data as Data

from .basewaveformtransformer import WaveformTransformer



class SingleChannelToyDenoiser(WaveformTransformer):
    
    def __init__(self, ms_before: float, ms_after: float, sampling_frequency: float):
        
        # Check model parameters match
        self.check_peak_interval_match(ms_before, ms_after, sampling_frequency)
        
        # Load model
        self.denoiser = self.load_model()
        
    def check_peak_interval_match(self, ms_before: float, ms_after: float, sampling_frequency: float):

        json_file_path = Path("/home/heberto/development/spikeinterface/bin/mearec_peak_interval.json")
        print(json_file_path.is_file())
        # Load the json file in the json_file_path_variable
        with open(json_file_path, "r") as json_file:
            peak_interval_dict = json.load(json_file)        
        
        assert peak_interval_dict["ms_before"] == ms_before
        assert peak_interval_dict["ms_after"] == ms_after
        assert peak_interval_dict["sampling_frequency"] == sampling_frequency
        
    def load_model(self):
            
        model_path = Path("/home/heberto/development/spikeinterface/bin/.marec.pt")
        denoiser = SingleChanDenoiser(pretrained_path=str(model_path), spike_size=128)
        denoiser = denoiser.load()

        return denoiser
        

    def transform(self, waveforms): 
        
        n_waveforms, n_timestamps, n_channels = waveforms.shape
        # Collapse channels and transform to torch tensor
        channelless_waveforms = waveforms.swapaxes(1, 2).reshape(-1, n_timestamps)
        waveforms_tensor = torch.from_numpy(channelless_waveforms).float()
        
        # Denoise
        denoised_channelless_waveforms = self.denoiser(waveforms_tensor)
        
        # Transform back to numpy
        denoised_waveforms_torch = denoised_channelless_waveforms.reshape(n_waveforms, n_channels, n_timestamps).swapaxes(2, 1)
        denoised_waveforms_numpy = denoised_waveforms_torch.detach().numpy() 
        
        return denoised_waveforms_numpy
    
    
class SingleChanDenoiser(nn.Module):
    def __init__(self, pretrained_path=None, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121):
        super().__init__()
        
        out_channels_conv1, out_channels_conv_2 = n_filters
        kernel_size_conv1, kernel_size_conv2 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, out_channels_conv1, kernel_size_conv1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels_conv1, out_channels_conv_2, kernel_size_conv2), nn.ReLU())
        n_input_feat = out_channels_conv_2 * (spike_size - kernel_size_conv1 - kernel_size_conv2 + 2)
        self.out = nn.Linear(n_input_feat, spike_size)
        self.pretrained_path = pretrained_path
    

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x

    #TODO: Put it outside the class
    def load(self, device="cpu"):
        checkpoint = torch.load(self.pretrained_path, map_location=device)
        self.load_state_dict(checkpoint)
        return self
    
