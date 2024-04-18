from __future__ import annotations

from pathlib import Path
import json
from typing import List, Optional

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
try:
    from huggingface_hub import hf_hub_download

    HAVE_HUGGINFACE = True
except ImportError:
    HAVE_HUGGINFACE = False

from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type
from .waveform_utils import to_temporal_representation, from_temporal_representation


class SingleChannelToyDenoiser(WaveformsNode):
    def __init__(
        self, recording: BaseRecording, return_output: bool = True, parents: Optional[List[PipelineNode]] = None
    ):
        assert HAVE_TORCH, "To use the SingleChannelToyDenoiser you need to install torch"
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"Model should have a {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        self.assert_model_and_waveform_temporal_match(waveform_extractor)

        # Load model
        self.denoiser = self.load_model()

    def assert_model_and_waveform_temporal_match(self, waveform_extractor: WaveformsNode):
        """
        Asserts that the model and the waveform extractor have the same temporal parameters
        """
        # Extract temporal parameters from the waveform extractor
        waveforms_ms_before = waveform_extractor.ms_before
        waveforms_ms_after = waveform_extractor.ms_after
        waveforms_sampling_frequency = waveform_extractor.recording.get_sampling_frequency()

        # Load the model temporal parameters
        repo_id = "SpikeInterface/test_repo"
        subfolder = "mearec_toy_model"
        filename = "params.json"

        json_file_path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename)
        # Load the json file in the json_file_path_variable
        with open(json_file_path, "r") as json_file:
            peak_interval_dict = json.load(json_file)

        model_ms_before = peak_interval_dict["ms_before"]
        model_ms_after = peak_interval_dict["ms_after"]
        model_sampling_frequency = peak_interval_dict["sampling_frequency"]

        ms_before_mismatch = waveforms_ms_before != model_ms_before
        ms_after_missmatch = waveforms_ms_after != model_ms_after
        sampling_frequency_mismatch = waveforms_sampling_frequency != model_sampling_frequency
        if ms_before_mismatch or ms_after_missmatch or sampling_frequency_mismatch:
            exception_string = (
                "Model and waveforms mismatch \n"
                f"{model_ms_before=} and {waveforms_ms_after=} \n"
                f"{model_ms_after=} and {waveforms_ms_after=} \n"
                f"{model_sampling_frequency=} and {waveforms_sampling_frequency=} \n"
            )
            raise ValueError(exception_string)

    def load_model(self):
        assert HAVE_HUGGINFACE, "To download models from Hugginface you need to install huggingface_hub"

        repo_id = "SpikeInterface/test_repo"
        subfolder = "mearec_toy_model"
        filename = "toy_model_marec.pt"

        model_path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename)
        denoiser = SingleChannel1dCNNDenoiser(pretrained_path=model_path, spike_size=128)
        denoiser = denoiser.load()

        return denoiser

    def compute(self, traces, peaks, waveforms):
        num_channels = waveforms.shape[2]

        # Collapse channels and transform to torch tensor
        temporal_waveforms = to_temporal_representation(waveforms)
        temporal_waveforms_tensor = torch.from_numpy(temporal_waveforms).float()

        # Denoise
        denoised_temporal_waveforms = self.denoiser(temporal_waveforms_tensor).detach().numpy()

        # Reconstruct representation with channels
        denoised_waveforms = from_temporal_representation(denoised_temporal_waveforms, num_channels)

        return denoised_waveforms


if HAVE_TORCH:

    class SingleChannel1dCNNDenoiser(nn.Module):
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

        def load(self, device="cpu"):
            checkpoint = torch.load(self.pretrained_path, map_location=device)
            self.load_state_dict(checkpoint)
            return self
