from pathlib import Path
import json 
from typing import List, Optional

import torch 
from torch import nn
from huggingface_hub import hf_hub_download


from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import PipelineNode, ExtractDenseWaveforms
from .waveform_utils import to_channelless_representation, from_channelless_representation


class SingleChannelToyDenoiser(PipelineNode):

    def __init__(self, recording: BaseRecording, ms_before: float, ms_after: float, sampling_frequency: float, 
                 return_ouput: bool = True, parents: Optional[List[PipelineNode ]] = None):
        super().__init__(recording, return_ouput=return_ouput, parents=parents)

        # Check model parameters match
        self.check_peak_interval_match(ms_before, ms_after, sampling_frequency)

        # Load model
        self.denoiser = self.load_model()
        
        self.assert_waveform_extractor_in_parents()

    def check_peak_interval_match(self, ms_before: float, ms_after: float, sampling_frequency: float):

        # Temporary
        repo_id = "SpikeInterface/test_repo"
        filename = "toy_model_peak_interval.json"

        json_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Load the json file in the json_file_path_variable
        with open(json_file_path, "r") as json_file:
            peak_interval_dict = json.load(json_file)        

        assert peak_interval_dict["ms_before"] == ms_before
        assert peak_interval_dict["ms_after"] == ms_after
        assert peak_interval_dict["sampling_frequency"] == sampling_frequency

    def load_model(self):
        repo_id = "SpikeInterface/test_repo"
        filename = "toy_model_marec.pt"

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        denoiser = PaninskiSingleChannelDenoiser(pretrained_path=model_path, spike_size=128)
        denoiser = denoiser.load()

        return denoiser


    def compute(self, traces, peaks, waveforms): 

        n_waveforms, n_timestamps, n_channels = waveforms.shape
        
        # Collapse channels and transform to torch tensor
        channelless_waveforms = to_channelless_representation(waveforms)
        channelless_waveforms_tensor = torch.from_numpy(channelless_waveforms).float()

        # Denoise
        denoised_channelless_waveforms = self.denoiser(channelless_waveforms_tensor).detach().numpy()
                
        # Reconstruct representation with channels
        desnoised_waveforms = from_channelless_representation(denoised_channelless_waveforms, n_channels)

        return desnoised_waveforms
    
    def assert_waveform_extractor_in_parents(self):
        
        if self.parents is None:
            raise ValueError("The SingleChannelToyDenoiser needs a WaveformExtractor as parent")
        else:
            some_parent_is_extractor = any((isinstance(parent, ExtractDenseWaveforms) for parent in self.parents))
            if not some_parent_is_extractor:
                raise ValueError("The SingleChannelToyDenoiser needs a WaveformExtractor as parent")


class PaninskiSingleChannelDenoiser(nn.Module):
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